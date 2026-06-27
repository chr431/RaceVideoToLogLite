from __future__ import annotations
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2

OCR_NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)?")


@dataclass
class SpeedObservation:
	timestamp: float
	raw_speed_kmh: float
	raw_text: str


def normalize_ocr_text(text: str) -> str:
	translation = str.maketrans(
		{
			"O": "0",
			"o": "0",
			"Q": "0",
			"D": "0",
			"I": "1",
			"l": "1",
			"|": "1",
			"!": "1",
			"Z": "2",
			"z": "2",
			"S": "5",
			"s": "5",
			"B": "8",
			"G": "6",
			"g": "6",
			"T": "7",
			"t": "7",
			",": ".",
		}
	)
	return text.translate(translation)


def extract_speed_value(ocr_result) -> tuple[float | None, str | None]:
	if not ocr_result:
		return None, None

	candidates: list[str] = []
	for item in ocr_result:
		if not item or len(item) < 2:
			continue
		text = str(item[1]).strip()
		if text:
			candidates.append(text)

	if not candidates:
		return None, None

	joined = normalize_ocr_text(" ".join(candidates)).replace(" ", "")
	match = OCR_NUMBER_RE.search(joined)
	if not match:
		return None, None

	raw_text = re.sub(r"\D", "", match.group(0))
	if not raw_text:
		return None, None
	try:
		return float(raw_text), raw_text
	except ValueError:
		return None, None


def ocr_digital_fallback(
	ocr, crop_bgr, max_speed_kmh=400
) -> tuple[float | None, str | None]:
	"""数字仪表 OCR 后备链：CLAHE+OTSU → 常规检测 → 无检测模式。

	用于 PP-OCR 标准预处理未命中时的后备策略（如赛车 HUD 仪表字体）。
	返回 (speed_value, raw_text) 或 (None, None)。
	"""
	gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

	# ── 策略1: CLAHE + OTSU + 常规检测 ──
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced = clahe.apply(gray)
	_, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	h, w = enhanced.shape[:2]
	for th in (28, 32, 48):
		scale = th / h
		resized = cv2.resize(enhanced, (max(1, int(w * scale)), th))
		bgr_input = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
		try:
			result, _ = ocr(bgr_input)
			sv, rt = extract_speed_value(result)
			if sv is not None and sv <= max_speed_kmh:
				return sv, rt
		except Exception:
			pass

	# ── 策略2: use_det=False（跳过检测，多预处理变体）──
	variants = [
		("clahe_otsu", enhanced),
		("inv", cv2.bitwise_not(gray)),
		("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
		("otsu_inv", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
	]
	for _label, img in variants:
		for th in (32, 48):
			scale = th / h
			resized = cv2.resize(img, (max(1, int(w * scale)), th))
			bgr_input = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
			try:
				result, _ = ocr(bgr_input, use_det=False)
				sv, rt = extract_speed_value(result)
				if sv is not None and sv <= max_speed_kmh:
					return sv, rt
			except Exception:
				pass

	return None, None


def clamp_region(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
	x1, y1, x2, y2 = roi
	x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
	y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
	return x1, y1, x2, y2


def build_speed_candidates(raw_text: str, max_speed_kmh: float) -> list[float]:
	"""根据 OCR 原始文本生成可能的速度候选值。

	策略:
	1. 数字后缀扩展: OCR "60" → 候选 60/160/260(处理丢位)
	2. 常见字符混淆替换: 6↔8, 3↔8, 5↔6, 0↔8, 1↔7 等
	"""
	if max_speed_kmh <= 0:
		return []

	text = re.sub(r"\D", "", raw_text)
	if not text:
		return []

	max_speed_int = int(math.floor(max_speed_kmh))
	if max_speed_int < 0:
		return []

	candidates: set[float] = set()

	# 策略1: 保留原始值
	try:
		val = int(text)
		if val <= max_speed_int:
			candidates.add(float(val))
	except ValueError:
		pass

	# 策略2: 后缀扩展（处理丢位）
	min_suffix_len = 1 if len(text) == 1 else max(1, len(text) - 2)
	for suffix_len in range(min_suffix_len, len(text) + 1):
		suffix_text = text[-suffix_len:]
		try:
			suffix_value = int(suffix_text)
		except ValueError:
			continue
		step = 10 ** suffix_len
		for candidate in range(suffix_value, max_speed_int + 1, step):
			candidates.add(float(candidate))

	# 策略3: 常见 OCR 字符混淆替换（对称映射）
	_CONFUSION_MAP = {
		"0": ["8", "6", "9"],
		"1": ["7", "2"],
		"2": ["7", "1", "3"],
		"3": ["8", "9", "2", "5"],
		"4": ["7", "9"],
		"5": ["6", "3", "8", "9"],
		"6": ["8", "5", "0", "2"],
		"7": ["1", "2", "4"],
		"8": ["0", "6", "3", "5", "9"],
		"9": ["8", "3", "5", "0", "4"],
	}
	for i, ch in enumerate(text):
		for alt in _CONFUSION_MAP.get(ch, []):
			altered = text[:i] + alt + text[i+1:]
			try:
				val = int(altered)
				if val <= max_speed_int:
					candidates.add(float(val))
			except ValueError:
				pass

	return sorted(candidates)


def _get_model_kwargs(variant: str, models_dir: str | None = None) -> dict | None:
	"""Get RapidOCR kwargs for the model. Returns None if files missing."""
	import rapidocr_onnxruntime as rr
	if models_dir is None:
		models_dir = str(Path(rr.__file__).parent / "models")
	cfg = {
		"det_model_path": f"{models_dir}/ch_PP-OCRv5_mobile_det_infer.onnx",
		"rec_model_path": f"{models_dir}/ch_PP-OCRv5_mobile_rec_infer.onnx",
		"text_score": 0.6, "use_angle_cls": False, "rec_batch_num": 12,
	}
	for key in ("det_model_path", "rec_model_path"):
		if not Path(cfg[key]).exists():
			return None
	_set_rec_keys_path(str(Path(rr.__file__).parent / "config.yaml"),
		f"{models_dir}/ppocr_keys_v1.txt")
	return cfg

def _set_rec_keys_path(config_path: str, keys_path: str) -> None:
	"""临时修改 rapidocr config.yaml 的 Rec.keys_path。"""
	from rapidocr_onnxruntime.utils import read_yaml
	config = read_yaml(config_path)
	if config.get("Rec", {}).get("keys_path") == keys_path:
		return  # 已设置
	config.setdefault("Rec", {})["keys_path"] = keys_path
	import yaml
	with open(config_path, "w") as f:
		yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def compute_video_hash(video_path: str | Path, chunk_size: int = 1_048_576) -> str:
	"""计算视频文件的快速哈希（头尾各 1MB + 文件大小）。

	使用 SHA-256，足以唯一标识视频文件，同时避免读取整个大文件。
	"""
	import hashlib
	video_path = Path(video_path)
	if not video_path.exists():
		return "N/A"
	file_size = video_path.stat().st_size
	h = hashlib.sha256()
	h.update(str(file_size).encode())
	with open(video_path, "rb") as f:
		h.update(f.read(chunk_size))
		if file_size > chunk_size * 2:
			f.seek(-chunk_size, 2)
			h.update(f.read(chunk_size))
	return h.hexdigest()[:16]  # 前 16 字符足够区分




def auto_select_anchors(observations, max_speed_kmh=400.0, window=0, max_dev=4.0):
	"""Select reliable OCR frames as Correction B anchors.

	Uses local median filter: for each frame, compute median in an adaptive
	sliding window. If frame value deviates <= max_dev from median, it is reliable.

	If window=0 (default), auto-computes window size to cover ~0.3s of data,
	making the filter robust at both high and low sampling rates.

	Returns set of trusted frame indices."""
	n = len(observations)
	raw_vals = [o.raw_speed_kmh for o in observations]
	anchors = set()

	# Adaptive window: cover ~0.3s regardless of sampling rate
	if window <= 0:
		times = [o.timestamp for o in observations]
		typical_dt = (times[-1] - times[0]) / max(n - 1, 1) if n > 1 else 0.017
		window = max(5, int(0.3 / max(typical_dt, 0.001)) | 1)  # odd, min 5
	half = window // 2

	for i in range(half, n - half):
		if raw_vals[i] <= 0:
			continue
		local = []
		for j in range(i - half, i + half + 1):
			if j != i and raw_vals[j] > 0 and raw_vals[j] <= max_speed_kmh:
				local.append(raw_vals[j])
		if len(local) < 3:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)

	# Head boundary frames
	for i in range(0, half):
		if raw_vals[i] <= 0:
			continue
		local = [raw_vals[j] for j in range(0, min(window, n))
		         if j != i and raw_vals[j] > 0 and raw_vals[j] <= max_speed_kmh]
		if len(local) < 2:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)

	# Tail boundary frames
	for i in range(n - half, n):
		if raw_vals[i] <= 0:
			continue
		local = [raw_vals[j] for j in range(max(0, n - window), n)
		         if j != i and raw_vals[j] > 0 and raw_vals[j] <= max_speed_kmh]
		if len(local) < 2:
			continue
		local.sort()
		median = local[len(local) // 2]
		if abs(raw_vals[i] - median) <= max_dev:
			anchors.add(i)

	# Post-filter: remove anchors that are extreme outliers vs immediate neighbors
	# An anchor must be within 10 km/h of at least one immediate neighbor
	anchors_filtered = set()
	for i in anchors:
		keep = True
		v = raw_vals[i]
		# Check against both neighbors
		left_ok = (i > 0 and raw_vals[i - 1] > 0 and abs(v - raw_vals[i - 1]) <= 10.0)
		right_ok = (i + 1 < n and raw_vals[i + 1] > 0 and abs(raw_vals[i + 1] - v) <= 10.0)
		# Keep if at least one neighbor is within 10 km/h
		if not left_ok and not right_ok:
			# Extreme outlier: not close to either neighbor
			keep = False
		if keep:
			anchors_filtered.add(i)

	return anchors_filtered

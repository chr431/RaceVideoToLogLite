"""Correction — 物理约束纠错流水线。

5 阶段流水线：错误检测 → 重OCR → 最优选择 → 多轮迭代 → 级联填充。
"""
from __future__ import annotations
import cv2
from ocr_engine import extract_speed_value, build_speed_candidates, ocr_digital_fallback

# 重 OCR 缓存（避免同一帧重复处理）
_reocr_cache: dict[int, set] = {}


def correct_with_anchors(rows, observations, raw_frames, ocr,
                         max_speed_kmh, max_accel_mps2, anchor_indices,
                         log_fn=None, progress_fn=None):
	"""5 阶段物理约束纠错流水线。

	以 anchor_indices 中帧的速度为硬约束（固定不变），
	对其余帧进行错误检测、重OCR、最优选择和级联填充。

	progress_fn(done, total): 滚动进度回调，在每个待修复帧处理完时调用。
	Returns: 修改后的 rows（原地修改）
	"""
	if len(anchor_indices) < 2:
		return rows

	n = len(rows)
	anchors = anchor_indices
	times = [r[0] for r in rows]

	if log_fn:
		log_fn(f"Correction: {n} rows, {len(anchors)} anchors")

	# ── 阶段 1：错误检测 ──
	error_set = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
	if log_fn:
		log_fn(f"  Stage 1: detected {len(error_set)} errors")
	if not error_set:
		return rows

	# ── 阶段 2+3：重 OCR + 最优选择（首轮）──
	fixed = _fix_errors(rows, observations, raw_frames, ocr, error_set,
	                    anchors, times, max_speed_kmh, max_accel_mps2,
	                    progress_fn=progress_fn)
	if log_fn:
		log_fn(f"  Stage 2+3: fixed {fixed} frames in round 1")

	# ── 阶段 4：多轮迭代 ──
	max_rounds = 3
	for rnd in range(2, max_rounds + 1):
		error_set = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		if not error_set:
			break
		fixed = _fix_errors(rows, observations, raw_frames, ocr, error_set,
		                    anchors, times, max_speed_kmh, max_accel_mps2,
		                    progress_fn=progress_fn)
		if log_fn:
			log_fn(f"  Stage 4 round {rnd}: {len(error_set)} errors, fixed {fixed}")

	# ── 阶段 5：迭代填充直到收敛（处理级联效应）──
	fill_pass = 0
	while fill_pass < 10:
		error_set = _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2)
		if not error_set:
			break
		_fill_unrecoverable(rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2,
		                    progress_fn=progress_fn)
		if log_fn:
			log_fn(f"  Stage 5 pass {fill_pass+1}: filled {len(error_set)} unrecoverable frames")
		fill_pass += 1

	return rows


def _detect_errors(rows, anchors, times, max_speed_kmh, max_accel_mps2):
	"""阶段 1：错误检测。6 种检测器并行标记异常帧。

	A. 邻帧跳变 — 与前后邻帧的加速度超限
	A2. V 字形 — 急减速后立即急加速（OCR 误读特征）
	A3. 悬崖 — 单侧极端跳变 + 对侧平坦
	B. 锚点趋势偏离 — 偏离锚点间线性插值过多
	C. 孤立离群 — 与两边都冲突但邻居彼此一致
	D. 局部趋势偏离 — 5 帧中位数偏离
	"""
	n = len(rows)
	raw_vals = [r[2] for r in rows]
	error_set = set()

	for i in range(n):
		if i in anchors:
			continue
		v = raw_vals[i]
		if v < 0 or v > max_speed_kmh:
			error_set.add(i)
			continue

		# ── A. 邻帧跳变检测 ──
		fwd_fail = False
		bwd_fail = False

		if i > 0:
			prev_v = raw_vals[i - 1]
			if prev_v >= 0 and prev_v <= max_speed_kmh:
				dt = max(times[i] - times[i - 1], 0.001)
				max_dv = max_accel_mps2 * dt * 3.6 * 1.2
				if abs(v - prev_v) > max_dv:
					if not (i + 1 < n and v == raw_vals[i + 1] and times[i + 1] - times[i] < 0.15):
						fwd_fail = True

		if i + 1 < n:
			next_v = raw_vals[i + 1]
			if next_v >= 0 and next_v <= max_speed_kmh:
				dt = max(times[i + 1] - times[i], 0.001)
				max_dv = max_accel_mps2 * dt * 3.6 * 1.2
				if abs(next_v - v) > max_dv:
					if not (i > 0 and v == raw_vals[i - 1] and times[i] - times[i - 1] < 0.15):
						bwd_fail = True

		if fwd_fail and bwd_fail:
			error_set.add(i)
			continue

		# ── A2. V 字形检测 ──
		if i > 0 and i + 1 < n:
			prev_v = raw_vals[i - 1]
			next_v = raw_vals[i + 1]
			if prev_v > 0 and next_v > 0:
				dt_left = max(times[i] - times[i - 1], 0.001)
				dt_right = max(times[i + 1] - times[i], 0.001)
				accel_left = (v - prev_v) / dt_left
				accel_right = (next_v - v) / dt_right
				accel_limit = max_accel_mps2 * 3.6 * 2.5
				if abs(accel_left) > accel_limit and accel_left * accel_right < 0:
					if not (i + 1 < n and v == raw_vals[i + 1] and times[i + 1] - times[i] < 0.15):
						error_set.add(i)
						continue
				if abs(accel_right) > accel_limit and accel_right * accel_left < 0:
					if not (i > 0 and v == raw_vals[i - 1] and times[i] - times[i - 1] < 0.15):
						error_set.add(i)
						continue

		# ── A3. 悬崖检测 ──
		if i > 0 and i + 1 < n:
			prev_v = raw_vals[i - 1]
			next_v = raw_vals[i + 1]
			if prev_v > 0 and next_v > 0:
				dt_left = max(times[i] - times[i - 1], 0.001)
				dt_right = max(times[i + 1] - times[i], 0.001)
				accel_left = (v - prev_v) / dt_left
				accel_right = (next_v - v) / dt_right
				cliff_limit = max_accel_mps2 * 3.6 * 3.0
				if abs(accel_left) > cliff_limit and abs(accel_right) < cliff_limit * 0.3:
					error_set.add(i)
					continue
				if abs(accel_right) > cliff_limit and abs(accel_left) < cliff_limit * 0.3:
					error_set.add(i)
					continue

		# ── B. 锚点趋势偏离 ──
		la = None; ra = None
		for j in range(i - 1, -1, -1):
			if j in anchors:
				la = j; break
		for j in range(i + 1, n):
			if j in anchors:
				ra = j; break
		if la is not None and ra is not None:
			lv = rows[la][2]; rv = rows[ra][2]
			lt = rows[la][0]; rt = rows[ra][0]
			total_dt = max(rt - lt, 0.001)
			frac = (times[i] - lt) / total_dt
			interp = lv + (rv - lv) * frac
			seg_dt = times[i] - lt
			threshold = max(5.0, 3.0 * max_accel_mps2 * max(seg_dt, 0.1) * 3.6)
			if abs(v - interp) > threshold:
				error_set.add(i)
				continue

		# ── C. 孤立离群 (spike) ──
		if i >= 2 and i + 2 < n:
			left_v = raw_vals[i - 1] if raw_vals[i - 1] >= 0 else (raw_vals[i - 2] if raw_vals[i - 2] >= 0 else None)
			right_v = raw_vals[i + 1] if raw_vals[i + 1] >= 0 else (raw_vals[i + 2] if raw_vals[i + 2] >= 0 else None)
			if left_v is not None and right_v is not None:
				dt_cross = max(times[i + 2] - times[i - 2], 0.01)
				max_dv_cross = max_accel_mps2 * dt_cross * 3.6 * 1.5
				if abs(right_v - left_v) <= max_dv_cross:
					dt_left = max(times[i] - times[i - 1], 0.001)
					dt_right = max(times[i + 1] - times[i], 0.001)
					max_dv_l = max_accel_mps2 * dt_left * 3.6 * 1.5
					max_dv_r = max_accel_mps2 * dt_right * 3.6 * 1.5
					if abs(v - left_v) > max_dv_l and abs(right_v - v) > max_dv_r:
						error_set.add(i)

		# ── D. 局部趋势偏离 ──
		if i >= 2 and i + 2 < n:
			window = []
			for j in range(max(0, i - 2), min(n, i + 3)):
				if j != i and raw_vals[j] >= 0 and raw_vals[j] <= max_speed_kmh:
					window.append(raw_vals[j])
			if len(window) >= 3:
				window.sort()
				local_median = window[len(window) // 2]
				dev = abs(v - local_median)
				if dev > 3.0:
					left_ok = (i >= 1 and raw_vals[i - 1] >= 0 and abs(raw_vals[i - 1] - local_median) < 2.0)
					right_ok = (i + 1 < n and raw_vals[i + 1] >= 0 and abs(raw_vals[i + 1] - local_median) < 2.0)
					if left_ok and right_ok:
						error_set.add(i)

	return error_set


def _fix_errors(rows, observations, raw_frames, ocr, error_set,
                anchors, times, max_speed_kmh, max_accel_mps2,
                progress_fn=None):
	"""阶段 2+3：对每个 error 帧重 OCR 获取备选，选最优值填入。"""
	fixed = 0
	progress_done = 0
	error_list = sorted(i for i in error_set if i not in anchors)
	total = len(error_list)
	for i in error_list:
		candidates = list(_re_ocr_frame(raw_frames[i][1], ocr, max_speed_kmh))
		interp_cand = _interp_candidate(i, rows, anchors, times, max_speed_kmh)
		if interp_cand is not None:
			candidates.append(interp_cand)
		oid = min(i, len(observations) - 1)
		confusion_cands = build_speed_candidates(observations[oid].raw_text, max_speed_kmh)
		candidates.extend(c for c in confusion_cands if c not in candidates)

		if candidates:
			raw_val = rows[i][2]
			reocr_unique = _re_ocr_frame(raw_frames[i][1], ocr, max_speed_kmh)

			# 若重 OCR 无法产生与原值不同的候选，且插值候选偏差 > 10 km/h，
			# 直接使用插值（信任物理模型优于 OCR）
			if len(reocr_unique) <= 1 and interp_cand is not None and abs(interp_cand - raw_val) > 10.0:
				if abs(raw_val - interp_cand) > 0.5:
					rows[i][2] = interp_cand
					if rows[i][3] == 0:
						rows[i][3] = 1
					fixed += 1
			else:
				best_val = None
				best_score = -1.0
				for cand in set(candidates):
					if not (0 <= cand <= max_speed_kmh):
						continue
					score = _score_candidate(cand, i, rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2)
					if score > best_score:
						best_score = score
						best_val = cand

				if best_val is not None and abs(rows[i][2] - best_val) > 0.5:
					rows[i][2] = best_val
					if rows[i][3] == 0:
						rows[i][3] = 1
					fixed += 1

		progress_done += 1
		if progress_fn:
			progress_fn(progress_done, total)
	return fixed


def _re_ocr_frame(crop_bgr, ocr, max_speed_kmh):
	"""阶段 2：对单帧尝试 4 种预处理变体重 OCR，有缓存。"""
	global _reocr_cache
	cache_key = hash(crop_bgr.tobytes()) if crop_bgr is not None and crop_bgr.size > 0 else None
	if cache_key is not None and cache_key in _reocr_cache:
		return _reocr_cache[cache_key]
	candidates = set()
	if crop_bgr is None or crop_bgr.size == 0:
		return candidates

	gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
	h, w = gray.shape[:2]
	if h <= 0 or w <= 0:
		return candidates

	def _do_ocr(img_bgr):
		res, _ = ocr(img_bgr)
		sv, rt = extract_speed_value(res)
		if sv is not None and sv <= max_speed_kmh:
			candidates.add(float(sv))

	# 变体 1: 标准灰度 (h=24)
	scale = 24.0 / h if h > 0 else 1.0
	proc = cv2.resize(gray, (max(1, int(w * scale)), 24))
	_do_ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))

	# 变体 2: CLAHE + OTSU (h=32)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	_, otsu = cv2.threshold(clahe.apply(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	scale32 = 32.0 / h if h > 0 else 1.0
	proc = cv2.resize(otsu, (max(1, int(w * scale32)), 32))
	_do_ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))

	# 变体 3: OTSU 反相 (h=32)
	_, otsu3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	proc = cv2.resize(otsu3, (max(1, int(w * scale32)), 32))
	_do_ocr(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))

	# 变体 4: ocr_digital_fallback 后备
	try:
		sv, _rt = ocr_digital_fallback(ocr, crop_bgr, max_speed_kmh)
		if sv is not None:
			candidates.add(float(sv))
	except Exception:
		pass

	if cache_key is not None:
		_reocr_cache[cache_key] = candidates
	return candidates


def _interp_candidate(i, rows, anchors, times, max_speed_kmh):
	"""计算帧 i 在左右锚点间的线性插值估计。"""
	n = len(rows)
	la = None; ra = None
	for j in range(i - 1, -1, -1):
		if j in anchors:
			la = j; break
	for j in range(i + 1, n):
		if j in anchors:
			ra = j; break
	if la is not None and ra is not None:
		lv = rows[la][2]; rv = rows[ra][2]
		lt = rows[la][0]; rt = rows[ra][0]
		total_dt = max(rt - lt, 0.001)
		frac = (times[i] - lt) / total_dt
		val = lv + (rv - lv) * frac
		if 0 <= val <= max_speed_kmh:
			return val
	return None


def _score_candidate(val, i, rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2):
	"""阶段 3：对候选值评分。

	score = neighbor_score * 0.4 + anchor_score * 0.35 + smoothness_score * 0.25
	"""
	n = len(rows)

	# 1. neighbor_score
	neighbor_score = 0.0
	count = 0
	for j in range(i - 1, max(i - 4, -1), -1):
		if j in error_set or rows[j][2] < 0 or rows[j][2] > max_speed_kmh:
			continue
		dt = max(times[i] - times[j], 0.001)
		max_dv = max_accel_mps2 * dt * 3.6
		dv = abs(val - rows[j][2])
		neighbor_score += 1.0 - dv / max(max_dv, 0.1) if dv <= max_dv else 0.0
		count += 1
		break
	for j in range(i + 1, min(i + 5, n)):
		if j in error_set or rows[j][2] < 0 or rows[j][2] > max_speed_kmh:
			continue
		dt = max(times[j] - times[i], 0.001)
		max_dv = max_accel_mps2 * dt * 3.6
		dv = abs(rows[j][2] - val)
		neighbor_score += 1.0 - dv / max(max_dv, 0.1) if dv <= max_dv else 0.0
		count += 1
		break
	neighbor_score = neighbor_score / max(count, 1)

	# 2. anchor_score
	anchor_score = 0.0
	interp = _interp_candidate(i, rows, anchors, times, max_speed_kmh)
	if interp is not None:
		dev = abs(val - interp)
		threshold = max(5.0, max_accel_mps2 * 3.6)
		anchor_score = max(0.0, 1.0 - dev / threshold)

	# 3. smoothness_score
	smoothness_score = 0.5
	if i >= 1 and i + 1 < n:
		prev_v = None
		for j in range(i - 1, max(i - 3, -1), -1):
			if j not in error_set and 0 <= rows[j][2] <= max_speed_kmh:
				prev_v = rows[j][2]; break
		next_v = None
		for j in range(i + 1, min(i + 4, n)):
			if j not in error_set and 0 <= rows[j][2] <= max_speed_kmh:
				next_v = rows[j][2]; break
		if prev_v is not None and next_v is not None:
			expected = (prev_v + next_v) / 2.0
			dev2 = abs(val - expected)
			smoothness_score = max(0.0, 1.0 - dev2 / max(10.0, max_accel_mps2 * 1.8 * 3.6))

	return neighbor_score * 0.4 + anchor_score * 0.35 + smoothness_score * 0.25


def _fill_unrecoverable(rows, anchors, error_set, times, max_speed_kmh, max_accel_mps2,
                        progress_fn=None):
	"""阶段 5：对无法通过重 OCR 修复的帧，从左到右传播可信值。"""
	n = len(rows)
	sorted_errors = sorted(i for i in error_set if i not in anchors)
	total = len(sorted_errors)
	progress_done = 0
	for i in sorted_errors:
		la = None
		for j in range(i - 1, -1, -1):
			if j in anchors or j not in error_set:
				if 0 <= rows[j][2] <= max_speed_kmh:
					la = j; break
		if la is not None:
			lv = rows[la][2]; lt = rows[la][0]
			ra = None
			for j in range(i + 1, n):
				if j in anchors:
					if 0 <= rows[j][2] <= max_speed_kmh:
						ra = j; break
			if ra is not None:
				rv = rows[ra][2]; rt = rows[ra][0]
				total_dt = max(rt - lt, 0.001)
				frac = (times[i] - lt) / total_dt
				val = lv + (rv - lv) * frac
			else:
				val = lv
			dt = max(times[i] - lt, 0.001)
			max_dv = max_accel_mps2 * dt * 3.6
			val = max(lv - max_dv, min(lv + max_dv, val))
			val = max(0.0, min(max_speed_kmh, val))
			rows[i][2] = val
			if rows[i][3] == 0:
				rows[i][3] = 1

		progress_done += 1
		if progress_fn:
			progress_fn(progress_done, total)

import streamlit as st
import pandas as pd
import csv
import io
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Set
from PIL import Image, ImageOps
import numpy as np
import math

# ✅ QRコード生成用（追加）
import qrcode

# ✅ スマホでの見やすさ改善（追加）
st.set_page_config(page_title="印刷用画像一括チェックアプリ v5.0", layout="centered")

# OpenCVのチェックはそのまま
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# DPI比較用の許容誤差 (厳密な一致にわずかな遊びを持たせる)
DPI_TOLERANCE = 1.0

# ----------------------------
# 📏 規格サイズ定義 (mm)
# ----------------------------
STANDARD_SIZES_MM: Dict[str, Tuple[float, float]] = {
    "指定なし": (0.0, 0.0),
    "A3 (297 x 420mm)": (297.0, 420.0),
    "A4 (210 x 297mm)": (210.0, 297.0),
    "A5 (148 x 210mm)": (148.0, 210.0),
    "B4 (257 x 364mm)": (257.0, 364.0),
    "B5 (182 x 257mm)": (182.0, 257.0),
}

# 🎯 定数: サイズチェックモード
SIZE_CHECK_MODE_BOTH = "両方が一致 (WとH)"
SIZE_CHECK_MODE_EITHER = "いずれかが一致 (WまたはH)"

# ----------------------------
# データクラスと解析関数の定義
# ----------------------------

@dataclass
class CheckConditions:
    required_color: Optional[str] = None
    required_dpi: Optional[int] = None
    min_width_mm: Optional[float] = None
    min_height_mm: Optional[float] = None
    require_trim: bool = False
    allowed_extensions: Optional[Set[str]] = None
    skip_all_checks: bool = False
    size_check_mode: str = SIZE_CHECK_MODE_BOTH
    allow_rotation: bool = False
    size_tolerance_mm: float = 0.1

@dataclass
class ImageReport:
    path: str
    width_px: int
    height_px: int
    dpi: Tuple[Optional[float], Optional[float]]
    width_mm: Optional[float]
    height_mm: Optional[float]
    mode: str
    color_family: str
    has_alpha: bool
    icc_profile: Optional[str]
    trim_marks_detected: Optional[bool]
    trim_marks_score: Optional[float]
    detected_extension: str
    notes: List[str]
    passed: bool = False

# ----------------------------------------------------------------------
# 🔧 ヘルパー関数群
# ----------------------------------------------------------------------

def _guess_color_family(img: Image.Image) -> str:
    m = img.mode.upper()
    if m in ("RGB", "RGBA"):
        try:
            arr = np.array(img)
            # グレースケールの簡易判定（RGBが全て同じ値）
            if arr.ndim == 3 and arr.shape[2] >= 3 and np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
                return "グレースケール"
        except Exception:
            pass
        return "RGB"
    elif m in ("CMYK", "CMYKA"):
        return "CMYK"
    elif m in ("L", "LA"):
        return "グレースケール"
    elif m == "P":
        return "インデックス"
    return m

def _get_dpi(img: Image.Image) -> Tuple[Optional[float], Optional[float]]:
    dpi = img.info.get("dpi", None)
    if isinstance(dpi, tuple) and len(dpi) == 2:
        try:
            return float(dpi[0]), float(dpi[1])
        except Exception:
            return None, None
    return None, None

def _mm_size(width_px: int, height_px: int, dpi: Tuple[Optional[float], Optional[float]]):
    xdpi, ydpi = dpi
    if xdpi and ydpi and xdpi > 0 and ydpi > 0:
        return (width_px / xdpi) * 25.4, (height_px / ydpi) * 25.4
    return None, None

def _detect_trim_marks(img: Image.Image) -> Tuple[Optional[bool], Optional[float]]:
    if not _HAS_CV2:
        return None, None
    try:
        g = np.array(img.convert("L"))
        h, w = g.shape
        margin_h, margin_w = max(5, int(h * 0.05)), max(5, int(w * 0.05))
        corners = [
            g[:margin_h, :margin_w],
            g[:margin_h, -margin_w:],
            g[-margin_h:, :margin_w],
            g[-margin_h:, -margin_w:]
        ]
        total_lines = 0
        for corner in corners:
            edges = cv2.Canny(corner, 150, 250)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=20, maxLineGap=5)
            if lines is not None:
                total_lines += len(lines)

        score = min(1.0, total_lines / 4.0)
        detected = score > 0.4
        return detected, score
    except Exception:
        return None, None

# ----------------------------------------------------------------------
# 🎯 サイズ比較関数 (許容誤差を引数で受け取る)
# ----------------------------------------------------------------------
def _check_size_match(
    det_w: Optional[float],
    det_h: Optional[float],
    req_w: Optional[float],
    req_h: Optional[float],
    cond: CheckConditions,
    tolerance_mm: float
) -> Tuple[bool, bool, bool, Optional[float], Optional[float]]:
    """
    画像の検出サイズと要求サイズを比較し、合格、幅OK、高さOKを返す。
    allow_rotation=True の場合、縦横の入れ替えを許容する。
    """
    if det_w is None or det_h is None:
        return False, False, False, None, None

    is_w_required = req_w is not None and req_w > 0
    is_h_required = req_h is not None and req_h > 0

    if not is_w_required and not is_h_required:
        return True, True, True, det_w, det_h

    # 1. 厳密チェック (縦横固定)
    match_w = (not is_w_required) or math.isclose(det_w, req_w, abs_tol=tolerance_mm)
    match_h = (not is_h_required) or math.isclose(det_h, req_h, abs_tol=tolerance_mm)

    final_det_w, final_det_h = det_w, det_h

    # 2. 縦横入れ替えチェック
    if cond.allow_rotation and ((is_w_required and not match_w) or (is_h_required and not match_h)):
        match_w_rot = (not is_w_required) or math.isclose(det_w, req_h, abs_tol=tolerance_mm)
        match_h_rot = (not is_h_required) or math.isclose(det_h, req_w, abs_tol=tolerance_mm)

        if is_w_required and is_h_required and match_w_rot and match_h_rot:
            match_w, match_h = True, True
            final_det_w, final_det_h = det_h, det_w
        elif not is_w_required or not is_h_required:
            pass

    # 3. 最終合否判定
    if cond.size_check_mode == SIZE_CHECK_MODE_BOTH:
        is_passed = match_w and match_h
    else:
        is_passed = match_w or match_h
        if is_w_required and not match_w:
            is_passed = False
        if is_h_required and not match_h:
            is_passed = False

    return is_passed, match_w, match_h, final_det_w, final_det_h


def analyze_image_file(img_data: io.BytesIO, file_name: str, cond: CheckConditions, default_dpi: Optional[float] = 300) -> Optional[ImageReport]:
    detected_ext = os.path.splitext(file_name)[1].lstrip('.').lower()

    try:
        img = Image.open(img_data)
    except Exception as e:
        return ImageReport(
            path=file_name, width_px=0, height_px=0, dpi=(None, None), width_mm=None, height_mm=None,
            mode="Error", color_family="Unknown", has_alpha=False, icc_profile=None,
            trim_marks_detected=None, trim_marks_score=None,
            notes=[f"画像を開くエラー: {e}"],
            detected_extension=detected_ext, passed=False
        )

    img = ImageOps.exif_transpose(img)
    w, h = img.size
    dpi = _get_dpi(img)

    # DPI情報がない場合、デフォルトDPIを使用
    if dpi == (None, None) and default_dpi:
        dpi = (default_dpi, default_dpi)

    width_mm, height_mm = _mm_size(w, h, dpi)
    family = _guess_color_family(img)
    has_alpha = "A" in img.mode
    icc = img.info.get("icc_profile")
    trim_detected, trim_score = _detect_trim_marks(img)
    notes = []
    passed = True

    # 形式チェック
    ext_passed = True
    if cond.allowed_extensions and not cond.skip_all_checks:
        if detected_ext not in cond.allowed_extensions:
            notes.append(f"形式: 不一致 (検出={detected_ext.upper()} / 許容={', '.join(cond.allowed_extensions).upper()}) -> ❌ 不合格")
            passed = False
            ext_passed = False
        else:
            notes.append(f"形式: 合致 (検出={detected_ext.upper()})")
    else:
        notes.append(f"形式: 指定なし (検出={detected_ext.upper()})")

    # 全チェックスキップ
    if cond.skip_all_checks:
        notes.append("📝 全ての条件チェックをスキップしました。")
        passed = ext_passed
    else:
        # カラーチェック
        if cond.required_color:
            if family != cond.required_color:
                notes.append(f"カラー: 不一致 (検出={family} / 指定={cond.required_color}) -> ❌ 不合格")
                passed = False
            else:
                notes.append(f"カラー: 合致 (検出={family})")
        else:
            notes.append(f"カラー: 指定なし (検出={family})")

        # DPI 厳密一致チェック
        dpi_value = dpi[0] if dpi[0] is not None else None
        if cond.required_dpi:
            if dpi_value is None or not math.isclose(dpi_value, cond.required_dpi, abs_tol=DPI_TOLERANCE):
                notes.append(f"DPI: 不一致/不明 (検出={'不明' if dpi_value is None else f'{dpi_value:.0f}'} vs 指定={cond.required_dpi:.0f}) -> ❌ 不合格")
                passed = False
            else:
                notes.append(f"DPI: 合格 (検出={dpi_value:.0f})")
        else:
            notes.append(f"DPI: 指定なし (検出={'不明' if dpi_value is None else f'{dpi_value:.0f}'})")

        # サイズ一致判定
        is_size_required = (cond.min_width_mm and cond.min_width_mm > 0) or (cond.min_height_mm and cond.min_height_mm > 0)
        size_passed = True

        if is_size_required:
            if width_mm is None or height_mm is None:
                notes.append("サイズ: DPI不足のためmm換算不可 -> ❌ 不合格")
                passed = False
                size_passed = False
            else:
                (size_passed, width_ok, height_ok, final_det_w, final_det_h) = _check_size_match(
                    width_mm, height_mm, cond.min_width_mm, cond.min_height_mm, cond, cond.size_tolerance_mm
                )

                if size_passed and cond.allow_rotation and final_det_w != width_mm:
                    notes.append(f"幅/高さ: 縦横を入れ替えて判定しました ({final_det_w:.1f}x{final_det_h:.1f}mm)。")
                    width_mm, height_mm = final_det_w, final_det_h

                if not size_passed:
                    req_w_str = f"{cond.min_width_mm:.1f}" if cond.min_width_mm else "Any"
                    req_h_str = f"{cond.min_height_mm:.1f}" if cond.min_height_mm else "Any"
                    notes.append(f"幅/高さ: 指定 ({req_w_str}x{req_h_str}mm, 許容誤差±{cond.size_tolerance_mm:.1f}mm) と不一致 -> ❌ 不合格")
                    passed = False
                else:
                    notes.append(f"幅/高さ: 合格 (検出={width_mm:.1f}x{height_mm:.1f}mm)")
        else:
            notes.append(f"幅/高さ: 指定なし (検出={'不明' if width_mm is None else f'W{width_mm:.1f} x H{height_mm:.1f}mm'})")

        # トンボチェック
        if cond.require_trim:
            if _HAS_CV2:
                if trim_detected is None:
                    notes.append("トンボ: 検出エラー -> ❌ 不合格")
                    passed = False
                elif not trim_detected:
                    notes.append(f"トンボ: 不検出 (スコア={trim_score:.2f}) -> ❌ 不合格")
                    passed = False
                else:
                    notes.append(f"トンボ: 検出 (スコア={trim_score:.2f})")
            else:
                dpi_value = dpi[0] if dpi[0] is not None else None
                is_print_ready = (
                    dpi_value is not None and dpi_value >= 300 and
                    family in ["CMYK", "グレースケール", "RGB"] and
                    icc is not None
                )
                if not is_print_ready:
                    notes.append("トンボ: 代替チェック - DPI/ICCプロファイル/カラーモードのいずれかが不足しており、トンボがない可能性が高い -> ❌ 不合格")
                    passed = False
                else:
                    notes.append("トンボ: 代替チェック - 印刷用メタデータは揃っています。トンボ検出はOpenCVが必要です。")
        else:
            if _HAS_CV2:
                if trim_detected:
                    notes.append(f"トンボ: 検出あり (スコア={trim_score:.2f})")
                else:
                    notes.append("トンボ: 検出なし")
            else:
                notes.append("トンボ: 未チェック (OpenCV未インストール)")

    report = ImageReport(
        path=file_name,
        width_px=w,
        height_px=h,
        dpi=dpi,
        width_mm=width_mm,
        height_mm=height_mm,
        mode=img.mode,
        color_family=family,
        has_alpha=has_alpha,
        icc_profile=icc,
        trim_marks_detected=trim_detected,
        trim_marks_score=trim_score,
        detected_extension=detected_ext,
        notes=notes,
        passed=passed
    )
    return report

# --- 装飾付き表示用ユーティリティ関数 ---
def _get_status_badge(text: str, is_passed: bool, condition_exist: bool = True) -> str:
    if not condition_exist:
        return f'<span style="padding: 2px 5px; border-radius: 3px; background-color: #f0f2f6; color: #555555;">{text}</span>'

    color = "white"
    if is_passed:
        bg_color = "#4CAF50"
        icon = "✅"
    else:
        bg_color = "#F44336"
        icon = "❌"

    return f'<span style="padding: 2px 5px; border-radius: 3px; background-color: {bg_color}; color: {color}; font-weight: bold;">{icon} {text}</span>'

def report_to_styled_dict(cond: CheckConditions, report: ImageReport) -> Dict:
    overall_badge = _get_status_badge("合格", True) if report.passed else _get_status_badge("不合格", False)

    ext_exist = bool(cond.allowed_extensions) and not cond.skip_all_checks
    ext_passed = not cond.allowed_extensions or report.detected_extension in cond.allowed_extensions
    ext_text = report.detected_extension.upper()
    ext_badge = _get_status_badge(ext_text, ext_passed, ext_exist)

    is_skipped = cond.skip_all_checks

    color_exist = bool(cond.required_color) and not is_skipped
    color_passed = not cond.required_color or report.color_family == cond.required_color
    color_text = report.color_family
    color_badge = _get_status_badge(color_text, color_passed, color_exist)

    dpi_exist = bool(cond.required_dpi) and not is_skipped
    dpi_det = report.dpi[0] if report.dpi[0] is not None else None

    dpi_passed = True
    if dpi_exist:
        if dpi_det is None or not math.isclose(dpi_det, cond.required_dpi, abs_tol=DPI_TOLERANCE):
            dpi_passed = False
        else:
            dpi_passed = True
    else:
        dpi_passed = True

    dpi_text = f"{dpi_det:.0f}" if dpi_det else "不明"
    dpi_badge = _get_status_badge(dpi_text, dpi_passed, dpi_exist)

    size_exist = ((cond.min_width_mm and cond.min_width_mm > 0) or (cond.min_height_mm and cond.min_height_mm > 0)) and not is_skipped
    size_passed = True
    size_text = ""

    current_det_w = report.width_mm
    current_det_h = report.height_mm

    if current_det_w is not None and current_det_h is not None:
        size_text = f"W{current_det_w:.1f} x H{current_det_h:.1f} mm"
        size_passed = True
        if size_exist:
            if any("幅/高さ: 指定" in n and "❌ 不合格" in n for n in report.notes):
                size_passed = False
        if is_skipped:
            size_passed = True
    else:
        if size_exist:
            size_passed = False
        size_text = "不明/DPI不足"

    size_badge = _get_status_badge(size_text, size_passed, size_exist)

    trim_exist = cond.require_trim and not is_skipped

    if trim_exist:
        is_alt_check_fail = False
        trim_text = ""

        if report.trim_marks_detected is None and not _HAS_CV2:
            is_print_ready = (
                dpi_det is not None and dpi_det >= 300 and
                report.color_family in ["CMYK", "グレースケール", "RGB"] and
                report.icc_profile is not None
            )
            if not is_print_ready:
                trim_passed = False
                trim_text = "メタデータ不足(要OpenCV)"
                is_alt_check_fail = True
            else:
                trim_passed = True
                trim_text = "メタデータOK (要OpenCV)"

        elif report.trim_marks_detected is None:
            trim_passed = False
            trim_text = "検出不可 (CV2エラー)"
        elif report.trim_marks_detected:
            trim_passed = True
            trim_text = f"あり ({report.trim_marks_score:.2f})"
        else:
            trim_passed = False
            trim_text = "なし"

        if is_alt_check_fail:
            trim_badge = _get_status_badge(trim_text, False, True)
        else:
            trim_badge = _get_status_badge(trim_text, trim_passed, trim_exist)

    else:
        trim_passed = True
        if report.trim_marks_detected is None:
            trim_text = "未チェック"
        elif report.trim_marks_detected:
            trim_text = f"検出あり ({report.trim_marks_score:.2f})"
        else:
            trim_text = "検出なし"
        trim_badge = _get_status_badge(trim_text, trim_passed, trim_exist)

    warning_notes = []
    if report.has_alpha:
        warning_notes.append("透過(α)あり")
    if not report.icc_profile:
        warning_notes.append("ICCなし")

    notes_text = " | ".join(report.notes)

    return {
        "ファイル名": report.path,
        "総合判定": overall_badge,
        "形式": ext_badge,
        "DPI": dpi_badge,
        "カラーモード": color_badge,
        "サイズ(mm)": size_badge,
        "トンボ": trim_badge,
        "注意": " / ".join(warning_notes) if warning_notes else "-",
        "詳細メモ": notes_text
    }

def generate_bulk_csv_data(reports: List[ImageReport]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "パス", "総合判定", "検出形式", "幅(px)", "高さ(px)", "DPI(x,y)", "幅(mm)", "高さ(mm)",
        "モード", "検出カラー", "透過", "ICCプロファイル",
        "トンボ検出", "トンボスコア", "詳細メモ"
    ])

    for report in reports:
        writer.writerow([
            report.path,
            "合格" if report.passed else "不合格",
            report.detected_extension.upper(),
            report.width_px,
            report.height_px,
            report.dpi,
            f"{report.width_mm:.1f}" if report.width_mm is not None else "",
            f"{report.height_mm:.1f}" if report.height_mm is not None else "",
            report.mode,
            report.color_family,
            "あり" if report.has_alpha else "なし",
            "あり" if report.icc_profile else "なし",
            "あり" if report.trim_marks_detected else ("なし" if report.trim_marks_detected is False else "検出不可"),
            f"{report.trim_marks_score:.3f}" if report.trim_marks_score is not None else "",
            " | ".join(report.notes)
        ])

    return ('\ufeff' + output.getvalue()).encode("utf-8")

# ----------------------------
# Streamlit アプリケーションのメイン
# ----------------------------

st.title("印刷用画像一括チェックアプリ v5.0 ")
st.caption(f"DPI比較許容誤差: **±{DPI_TOLERANCE}dpi**")

# ✅ 公開URLが設定されていればQR表示（追加）
# Streamlit CloudでSecretsに PUBLIC_URL="https://xxxxx.streamlit.app" を入れると表示される
with st.expander("📱 スマホで開く（QRコード）", expanded=True):
    public_url = st.secrets.get("PUBLIC_URL", "")
    if public_url:
        qr_img = qrcode.make(public_url)
        st.image(qr_img, caption="スマホでQRを読み込むとアプリが開きます", use_container_width=False)
        st.write(public_url)
    else:
        st.info("まだ公開URLが設定されていません。Streamlit CloudのSecretsに PUBLIC_URL を設定すると、ここにQRが出ます。")

# --- 印刷条件設定 (サイドバーに移動) ---
st.sidebar.header("設定")
st.sidebar.markdown("---")

# **🎯 全チェックスキップオプション**
skip_all = st.sidebar.checkbox(
    "📝 全ての条件チェックを無効にする",
    value=False,
    help="このチェックを有効にすると、DPI、サイズ、カラー、トンボの判定は行われず、形式チェックのみが適用されます。"
)
st.sidebar.markdown("---")

# 対象ファイル形式選択
ALL_SUPPORTED_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "pdf", "gif", "bmp", "webp"]

selected_types = st.sidebar.multiselect(
    "🖼️ 許容するファイル形式 (検出形式の合否判定に使用)",
    options=ALL_SUPPORTED_TYPES,
    default=["jpg", "jpeg", "png", "tif", "tiff"]
)
if 'pdf' in selected_types:
    st.sidebar.warning("⚠️ PDFの解析は環境設定に依存します。")

# --- 条件設定 ---
st.sidebar.subheader("条件設定 (無効にしない場合)")

if skip_all:
    st.sidebar.markdown("> すべての条件チェックは無効です。")
    default_w, default_h = 0.0, 0.0
    color, required_dpi, min_width, min_height, trim = None, None, 0.0, 0.0, False
    size_mode = SIZE_CHECK_MODE_BOTH
    allow_rot = False
    size_tolerance = 0.1
else:
    size_preset = st.sidebar.selectbox(
        "📏 規格サイズを選択 (指定サイズ)",
        options=list(STANDARD_SIZES_MM.keys())
    )
    default_w, default_h = STANDARD_SIZES_MM.get(size_preset, (0.0, 0.0))

    color = st.sidebar.selectbox(
        "求めるカラーモード",
        options=[None, "CMYK", "RGB", "グレースケール"],
        format_func=lambda x: x if x else "指定なし"
    )

    required_dpi_input = st.sidebar.number_input(
        "指定DPI (厳密一致)",
        min_value=0,
        value=300,
        step=10,
        format="%d",
        help="画像に設定されているDPIと厳密に一致することを求めます。0の場合はチェックしません。",
        placeholder="300"
    )
    required_dpi = required_dpi_input if required_dpi_input > 0 else None

    size_tolerance = st.sidebar.number_input(
        "サイズ許容誤差(mm) (計算誤差対策)",
        min_value=0.0,
        value=0.1,
        step=0.01,
        format="%.2f",
        help="指定サイズ(mm)との間に許容する誤差です。計算誤差を避けるため0.1mmを推奨します。"
    )

    min_width = st.sidebar.number_input("指定幅(mm)", min_value=0.0, value=default_w, step=1.0, format="%.1f")
    min_height = st.sidebar.number_input("指定高さ(mm)", min_value=0.0, value=default_h, step=1.0, format="%.1f")

    allow_rot = st.sidebar.checkbox(
        "🔄 縦横を自動で合わせる (幅と高さを入れ替えても合格とする)",
        value=True,
        help="例: A4横(W297, H210)を指定した場合、画像がA4縦(W210, H297)でも合格になります。片方のみ指定の場合は無効です。"
    )

    size_mode = st.sidebar.selectbox(
        "幅と高さのチェック方法",
        options=[SIZE_CHECK_MODE_BOTH, SIZE_CHECK_MODE_EITHER],
        index=0,
        help="『両方が一致』はWとHの両方が指定サイズ±誤差に一致する必要があります。"
    )

    trim = st.sidebar.checkbox("トンボ必須", value=False)

# --- 条件クラスの生成 ---
cond = CheckConditions(
    required_color=color,
    required_dpi=required_dpi,
    min_width_mm=min_width if min_width > 0 else None,
    min_height_mm=min_height if min_height > 0 else None,
    require_trim=trim,
    allowed_extensions=set(selected_types),
    skip_all_checks=skip_all,
    size_check_mode=size_mode,
    allow_rotation=allow_rot,
    size_tolerance_mm=size_tolerance
)

st.sidebar.info(f"OpenCV: {'✅ インストール済み' if _HAS_CV2 else '❌ 未インストール (トンボ検出不可/代替判定)'}")
st.sidebar.markdown("---")

# ✅ サイドバーにもQR（追加）
st.sidebar.subheader("📱 スマホで開く（QR）")
public_url_sidebar = st.secrets.get("PUBLIC_URL", "")
if public_url_sidebar:
    qr_img2 = qrcode.make(public_url_sidebar)
    st.sidebar.image(qr_img2, caption="QRでアクセス", use_container_width=True)
else:
    st.sidebar.caption("SecretsでPUBLIC_URLを設定するとQRが出ます")

# --- 画像ファイルのアップロード ---
st.header("1. 画像ファイルのアップロード")
uploaded_files = st.file_uploader(
    "チェックするファイルを選択してください (複数選択可)",
    type=None,
    accept_multiple_files=True
)

if uploaded_files:
    st.header("2. 解析と結果")

    latest_iteration = st.empty()
    bar = st.progress(0)

    reports: List[ImageReport] = []

    for i, uploaded_file in enumerate(uploaded_files):
        progress_percentage = (i + 1) / len(uploaded_files)
        latest_iteration.text(f"処理中: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        bar.progress(progress_percentage)

        report = analyze_image_file(io.BytesIO(uploaded_file.getvalue()), uploaded_file.name, cond, default_dpi=300)

        if report:
            reports.append(report)

    bar.empty()
    latest_iteration.empty()

    if reports:
        st.success(f"✅ 全 {len(reports)} 件のファイル解析が完了しました。")

        st.subheader("総合判定一覧 (項目別合否)")

        styled_data = [report_to_styled_dict(cond, r) for r in reports]
        df_styled_summary = pd.DataFrame(styled_data)

        passed_count = sum(r.passed for r in reports)
        st.info(f"総合合格数: **{passed_count}** / 全体数: **{len(reports)}**")

        # Markdownテーブル（HTMLバッジ表示のため）
        columns = df_styled_summary.columns.tolist()
        markdown_table = " | " + " | ".join(columns) + " |\n"
        markdown_table += " | " + " | ".join(["---"] * len(columns)) + " |\n"
        for _, row in df_styled_summary.iterrows():
            markdown_table += " | " + " | ".join(row.astype(str)) + " |\n"

        st.markdown(markdown_table, unsafe_allow_html=True)

        # CSV一括ダウンロード
        csv_data = generate_bulk_csv_data(reports)
        st.download_button(
            label="🔽 全ファイルの結果をCSVでダウンロード",
            data=csv_data,
            file_name="bulk_file_check_report_v5_0_fixed.csv",
            mime="text/csv",
            type="primary"
        )

        st.markdown("---")
        st.subheader("個別詳細とプレビュー")

        for uploaded_file in uploaded_files:
            report = next((r for r in reports if r.path == uploaded_file.name), None)

            if report:
                with st.expander(f"**{'✅' if report.passed else '❌'} {report.path} の詳細**", expanded=False):
                    if report.mode not in ["Error", "Unknown"]:
                        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
                    else:
                        st.warning(f"ファイルモード: {report.mode}。プレビューはできません。")

                    st.markdown("#### 判定概要")
                    st.markdown(f"""
                    - **総合判定**: **{'✅ 合格' if report.passed else '❌ 不合格'}**
                    - **検出形式**: `{report.detected_extension.upper()}`
                    - **ピクセルサイズ**: `{report.width_px} x {report.height_px}`
                    - **DPI**: `{'不明' if report.dpi[0] is None else f'{report.dpi[0]:.0f}'}` (x) / `{'不明' if report.dpi[1] is None else f'{report.dpi[1]:.0f}'}` (y)
                    - **検出カラー**: `{report.color_family}`
                    """)

                    st.markdown("#### 詳細メモ (問題点・注意点)")
                    if report.notes:
                        note_markdown = "\n".join([
                            f"- {n.replace('-> ❌ 不合格', '<span style=\"color:red;\">❌ 不合格</span>')}"
                            for n in report.notes
                        ])
                        st.markdown(note_markdown, unsafe_allow_html=True)
                    else:
                        st.markdown("- 特記事項なし")
            else:
                st.warning(f"{uploaded_file.name} は解析中にエラーが発生しました。")
    else:
        st.error("アップロードされたファイルで解析に成功したものはありませんでした。ファイル形式を確認してください。")

else:
    st.info("ファイルをアップロードしてチェックを開始してください。")

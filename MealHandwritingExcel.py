import io
import os
import re
import time
from typing import List
import fitz  # PyMuPDF
from google import genai
from google.genai import errors, types
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
import streamlit as st


# ==========================================
# 1. AI 출력 데이터 규격 정의 (엄격한 스키마)
# ==========================================
class MealRow(BaseModel):
    date: str = Field(
        description="날짜 (예: 11/14). 비어있거나 위 행과 같아 생략/화살표 처리된 경우 해당 날짜를 채움"
    )
    name: str = Field(description="이름 (한글 성명)")
    breakfast_menu: str = Field(
        default="", description="조식 메뉴명 (없으면 빈 문자열)"
    )
    breakfast_price: int = Field(
        default=0, description="조식 금액 정수 (없으면 0)"
    )
    lunch_menu: str = Field(
        default="", description="중식 메뉴명 (없으면 빈 문자열)"
    )
    lunch_price: int = Field(default=0, description="중식 금액 정수 (없으면 0)")
    dinner_menu: str = Field(
        default="", description="석식 메뉴명 (없으면 빈 문자열)"
    )
    dinner_price: int = Field(
        default=0, description="석식 금액 정수 (없으면 0)"
    )
    ai_uncertain: bool = Field(
        default=False,
        description="글씨가 심하게 흘려 쓰여 판독이 불확실하면 True",
    )
    uncertain_note: str = Field(
        default="", description="불확실한 항목과 이유 (예: 중식 금액 8000인지 9000인지 모호)"
    )


class PageExtraction(BaseModel):
    rows: List[MealRow]


# ==========================================
# 2. 데이터 무결성 자동 검증 로직
# ==========================================
def validate_row(row: dict) -> tuple[str, str]:
    """각 행의 날짜, 이름, 메뉴-금액 짝, 15,000원 상한선을 규칙 기반으로 검증합니다."""
    issues = []

    if row.get("ai_uncertain"):
        issues.append(f"AI판독모호({row.get('uncertain_note', '육안확인필요')})")

    date_str = str(row.get("date", "")).strip()
    if not re.match(r"^\d{1,2}/\d{1,2}$", date_str):
        issues.append(f"날짜형식오류('{date_str}')")
    else:
        m, d = map(int, date_str.split("/"))
        if not (1 <= m <= 12 and 1 <= d <= 31):
            issues.append(f"날짜범위오류('{date_str}')")

    name_str = str(row.get("name", "")).strip()
    if not name_str or re.search(r"\d", name_str):
        issues.append(f"이름확인필요('{name_str}')")

    meals = [
        ("조식", row.get("breakfast_menu", ""), row.get("breakfast_price", 0)),
        ("중식", row.get("lunch_menu", ""), row.get("lunch_price", 0)),
        ("석식", row.get("dinner_menu", ""), row.get("dinner_price", 0)),
    ]

    for meal_name, menu, price in meals:
        menu = str(menu).strip() if menu is not None else ""
        try:
            price = int(price) if price is not None else 0
        except (ValueError, TypeError):
            issues.append(f"{meal_name}금액숫자아님")
            continue

        if price < 0:
            issues.append(f"{meal_name}음수금액")
        if price > 15000:
            issues.append(f"{meal_name}상한초과({price:,}원 > 15,000원)")
        if price > 0 and price % 100 != 0:
            issues.append(f"{meal_name}100원단위아님({price:,}원)")
        if menu and price == 0:
            issues.append(f"{meal_name}금액누락")
        if not menu and price > 0:
            issues.append(f"{meal_name}메뉴누락")

    if issues:
        return "⚠️ 확인필요", " / ".join(issues)
    return "✅ 정상", ""


# ==========================================
# 3. PDF -> 고속 처리용 최적화 이미지 변환
# ==========================================
def pdf_to_images(pdf_bytes: bytes, dpi: int = 160) -> List[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("jpg", jpg_quality=85)))
        if max(img.size) > 1600:
            img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        images.append(img)
    return images


# ==========================================
# 4. 사용 가능한 Gemini 모델 자동 탐색 및 우회 실행
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def get_available_models(api_key: str) -> List[str]:
    """현재 API 키로 호출 가능한 Gemini 비전 모델 목록을 구글 서버에서 직접 가져옵니다."""
    fallback_defaults = ["gemini-3.8-flash", "gemini-3.8-pro"]
    if not api_key:
        return fallback_defaults
    try:
        client = genai.Client(api_key=api_key)
        models = []
        for m in client.models.list():
            name = getattr(m, "name", "").replace("models/", "")
            actions = getattr(m, "supported_actions", []) or []
            if (
                "gemini" in name
                and "generateContent" in actions
                and not any(
                    x in name
                    for x in ["embedding", "imagen", "veo", "tts", "audio", "robotics"]
                )
            ):
                models.append(name)

        # flash 계열을 우선순위로 정렬
        flash_models = sorted([m for m in models if "flash" in m], reverse=True)
        other_models = sorted([m for m in models if "flash" not in m], reverse=True)
        combined = flash_models + other_models
        return combined if combined else fallback_defaults
    except Exception:
        return fallback_defaults


def extract_handwriting_from_image(
    api_key: str,
    image: Image.Image,
    candidate_models: List[str],
    status_box=None,
) -> List[dict]:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120_000),
    )

    prompt = """
    첨부된 스캔 이미지의 식대 장부 표를 위에서 아래로 한 행씩 읽어 JSON으로 추출하세요.
    열 순서: 날짜 | 이름 | 조식(메뉴, 금액) | 중식(메뉴, 금액) | 석식(메뉴, 금액)

    규칙:
    1. 날짜는 '11/14' 형태로 기록. 빈칸이나 '〃' 표시는 윗줄 날짜를 채움.
    2. 빈 끼니는 메뉴 "", 금액 0으로 출력.
    3. 금액은 순수 정수(예: 9500)로 변환. 한 끼 최대 15,000원 이하.
    4. 글씨가 불확실하면 ai_uncertain=true, uncertain_note에 사유 기재.
    5. 완전히 비어 있는 빈 행은 제외.
    """

    last_error = None
    # 최대 2바퀴 동안 후보 모델들을 순회하며 503/504 과부하 시 자동 전환
    for round_idx in range(1, 3):
        for model_name in candidate_models:
            try:
                config_kwargs = {
                    "response_mime_type": "application/json",
                    "response_schema": PageExtraction,
                    "temperature": 0.0,
                }
                if "flash" in model_name.lower():
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=0
                    )

                response = client.models.generate_content(
                    model=model_name,
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                parsed = PageExtraction.model_validate_json(response.text)
                return [row.model_dump() for row in parsed.rows]

            except (errors.ServerError, errors.ClientError) as e:
                last_error = e
                err_str = str(e)
                # 503(과부하), 504(시간초과), 429(한도초과), 404(미지원) 시 다음 모델로 즉시 우회
                if any(code in err_str for code in ["503", "504", "429", "404"]):
                    if status_box:
                        status_box.warning(
                            f"⚠️ `{model_name}` 서버 혼잡/제한으로 다른 모델로 자동 전환합니다..."
                        )
                    time.sleep(2)
                    continue
                raise e

        if status_box:
            status_box.warning(
                "⏳ 구글 AI 서버 전체에 일시적 트래픽이 몰려 10초 대기 후 다시 시도합니다..."
            )
        time.sleep(10)

    raise last_error


# ==========================================
# 5. 요청하신 2단 병합 헤더 엑셀(.xlsx) 생성
# ==========================================
def _safe_int(val) -> int:
    num = pd.to_numeric(val, errors="coerce")
    return 0 if pd.isna(num) else int(num)


def _safe_str(val) -> str:
    return "" if pd.isna(val) else str(val).strip()


def create_formatted_excel(df: pd.DataFrame) -> bytes:
    df = df.reset_index(drop=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "식대정리내역"

    thin_side = Side(border_style="thin", color="000000")
    border_all = Border(
        left=thin_side, right=thin_side, top=thin_side, bottom=thin_side
    )
    header_fill = PatternFill(
        start_color="F2F2F2", end_color="F2F2F2", fill_type="solid"
    )
    total_fill = PatternFill(
        start_color="FFF9E6", end_color="FFF9E6", fill_type="solid"
    )
    font_header = Font(name="맑은 고딕", size=11, bold=True)
    font_body = Font(name="맑은 고딕", size=10)
    font_total = Font(name="맑은 고딕", size=10, bold=True)
    align_center = Alignment(horizontal="center", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")

    ws.merge_cells("A1:A2")
    ws["A1"] = "날짜"

    ws.merge_cells("B1:B2")
    ws["B1"] = "이름"

    ws.merge_cells("C1:D1")
    ws["C1"] = "조식"
    ws["C2"] = "메뉴"
    ws["D2"] = "금액"

    ws.merge_cells("E1:F1")
    ws["E1"] = "중식"
    ws["E2"] = "메뉴"
    ws["F2"] = "금액"

    ws.merge_cells("G1:H1")
    ws["G1"] = "석식"
    ws["G2"] = "메뉴"
    ws["H2"] = "금액"

    for r in range(1, 3):
        ws.row_dimensions[r].height = 24
        for c in range(1, 9):
            cell = ws.cell(row=r, column=c)
            cell.font = font_header
            cell.alignment = align_center
            cell.border = border_all
            cell.fill = header_fill

    start_row = 3
    for idx, row in df.iterrows():
        r = start_row + idx
        ws.row_dimensions[r].height = 22

        b_price = _safe_int(row.get("조식_금액", 0))
        l_price = _safe_int(row.get("중식_금액", 0))
        d_price = _safe_int(row.get("석식_금액", 0))

        values = [
            _safe_str(row.get("날짜", "")),
            _safe_str(row.get("이름", "")),
            _safe_str(row.get("조식_메뉴", "")),
            b_price if b_price > 0 else "",
            _safe_str(row.get("중식_메뉴", "")),
            l_price if l_price > 0 else "",
            _safe_str(row.get("석식_메뉴", "")),
            d_price if d_price > 0 else "",
        ]

        for c_idx, val in enumerate(values, start=1):
            cell = ws.cell(row=r, column=c_idx, value=val)
            cell.font = font_body
            cell.border = border_all
            if c_idx in (4, 6, 8):
                cell.alignment = align_right
                if isinstance(val, int):
                    cell.number_format = "#,##0"
            else:
                cell.alignment = align_center

    end_row = start_row + len(df) - 1
    if len(df) > 0:
        total_row = end_row + 1
        ws.row_dimensions[total_row].height = 24
        ws.merge_cells(f"A{total_row}:B{total_row}")
        ws[f"A{total_row}"] = "합계"
        ws[f"D{total_row}"] = f"=SUM(D{start_row}:D{end_row})"
        ws[f"F{total_row}"] = f"=SUM(F{start_row}:F{end_row})"
        ws[f"H{total_row}"] = f"=SUM(H{start_row}:H{end_row})"

        for c_idx in range(1, 9):
            cell = ws.cell(row=total_row, column=c_idx)
            cell.font = font_total
            cell.border = border_all
            cell.fill = total_fill
            if c_idx in (4, 6, 8):
                cell.alignment = align_right
                cell.number_format = "#,##0"
            else:
                cell.alignment = align_center

    col_widths = {
        "A": 12,
        "B": 14,
        "C": 20,
        "D": 13,
        "E": 20,
        "F": 13,
        "G": 20,
        "H": 13,
    }
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


# ==========================================
# 6. Streamlit 웹 UI (좌우 원본 대조 및 검수)
# ==========================================
st.set_page_config(page_title="손글씨 식대 장부 엑셀 변환기", layout="wide")
st.title("📋 손글씨 식대 장부 → 엑셀 정밀 변환기")

with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
    )

    available_models = get_available_models(api_key)
    model_choice = st.selectbox(
        "우선 인식 모델 (과부하 시 자동 우회)",
        available_models,
        help="선택한 모델이 503 과부하일 경우 목록의 다른 모델로 자동 전환됩니다.",
    )

    st.info(
        "**자동 검증 기준**\n"
        "- 날짜 형식: `11/14` 형태\n"
        "- 끼니별 최대 금액: `15,000원`\n"
        "- 금액 단위: `100원` 단위\n"
        "- 메뉴·금액 짝 누락 여부"
    )

uploaded_pdf = st.file_uploader(
    "스캔한 PDF 파일을 업로드하세요", type=["pdf"]
)

if uploaded_pdf is not None:
    if (
        "pdf_name" not in st.session_state
        or st.session_state.pdf_name != uploaded_pdf.name
    ):
        st.session_state.pdf_name = uploaded_pdf.name
        st.session_state.images = pdf_to_images(uploaded_pdf.read(), dpi=160)
        st.session_state.extracted_df = None

    images = st.session_state.images
    st.success(f"총 **{len(images)}페이지**가 최적화되어 로드되었습니다.")

    if st.button(
        "🔍 손글씨 판독 및 무결성 검증 시작",
        type="primary",
        disabled=not api_key,
    ):
        # 사용자가 고른 모델을 맨 앞에 두고, 나머지 가용 모델을 백업 후보로 배치
        candidate_models = [model_choice] + [
            m for m in available_models if m != model_choice
        ]

        all_rows = []
        progress = st.progress(0)
        status_text = st.empty()
        for idx, img in enumerate(images):
            status_text.info(f"⏳ {idx + 1} / {len(images)} 페이지 분석 중...")
            page_rows = extract_handwriting_from_image(
                api_key, img, candidate_models, status_box=status_text
            )
            for r in page_rows:
                status, reason = validate_row(r)
                all_rows.append({
                    "페이지": idx + 1,
                    "검증상태": status,
                    "확인사유": reason,
                    "날짜": r["date"],
                    "이름": r["name"],
                    "조식_메뉴": r["breakfast_menu"],
                    "조식_금액": r["breakfast_price"],
                    "중식_메뉴": r["lunch_menu"],
                    "중식_금액": r["lunch_price"],
                    "석식_메뉴": r["dinner_menu"],
                    "석식_금액": r["dinner_price"],
                })
            progress.progress((idx + 1) / len(images))

        status_text.success("✅ 모든 페이지 분석이 완료되었습니다!")
        st.session_state.extracted_df = pd.DataFrame(all_rows)

    if st.session_state.get("extracted_df") is not None:
        st.divider()
        col_left, col_right = st.columns([1, 1.4])

        with col_left:
            st.subheader("📄 원본 스캔 이미지 대조")
            page_num = st.number_input(
                "확인할 페이지 번호",
                min_value=1,
                max_value=len(images),
                value=1,
                step=1,
            )
            st.image(images[page_num - 1], use_container_width=True)

        with col_right:
            st.subheader("✏️ 추출 결과 검수 및 수정")
            st.caption(
                "⚠️ 표시가 뜬 행을 원본과 비교해 직접 수정하세요. (행 추가/삭제 가능)"
            )

            edited_df = st.data_editor(
                st.session_state.extracted_df,
                num_rows="dynamic",
                use_container_width=True,
                height=520,
            )

            b_sum = pd.to_numeric(
                edited_df["조식_금액"], errors="coerce"
            ).fillna(0).sum()
            l_sum = pd.to_numeric(
                edited_df["중식_금액"], errors="coerce"
            ).fillna(0).sum()
            d_sum = pd.to_numeric(
                edited_df["석식_금액"], errors="coerce"
            ).fillna(0).sum()
            total_sum = b_sum + l_sum + d_sum

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("조식 합계", f"{int(b_sum):,}원")
            m2.metric("중식 합계", f"{int(l_sum):,}원")
            m3.metric("석식 합계", f"{int(d_sum):,}원")
            m4.metric("총 합계", f"{int(total_sum):,}원")

            excel_bytes = create_formatted_excel(edited_df)
            st.download_button(
                label="📥 양식 적용된 엑셀(.xlsx) 다운로드",
                data=excel_bytes,
                file_name="식대정리_완료.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
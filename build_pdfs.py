"""
Build fraudwatch_PRD.pdf and fraudwatch_case_study.pdf
"""
import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as pdfcanvas

# ── Palette ─────────────────────────────────────────────────────────────────
DARK    = colors.HexColor("#1a1a18")
ORANGE  = colors.HexColor("#D85A30")
GREEN   = colors.HexColor("#0F6E56")
PURPLE  = colors.HexColor("#534AB7")
BG      = colors.HexColor("#F5F5F0")
MID     = colors.HexColor("#E8E8E2")
MUTED   = colors.HexColor("#888880")
WHITE   = colors.white
ROW_ALT = colors.HexColor("#F0EFE9")

# ── Reusable style helpers ───────────────────────────────────────────────────

def style(name, **kw):
    return ParagraphStyle(name, **kw)

def divider(color=ORANGE, thickness=1.5, spaceBefore=10, spaceAfter=10):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=spaceAfter, spaceBefore=spaceBefore)

def sp(n=6):
    return Spacer(1, n)

class ColorBlock(Flowable):
    """Full-width colored rectangle background for section headers."""
    def __init__(self, text, bg=DARK, fg=WHITE, height=26, font_size=11):
        super().__init__()
        self.text = text
        self.bg = bg
        self.fg = fg
        self.block_height = height
        self.font_size = font_size
        self.width = 0  # set by wrap

    def wrap(self, avail_width, avail_height):
        self.width = avail_width
        return avail_width, self.block_height + 6

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.rect(0, 0, self.width, self.block_height, fill=1, stroke=0)
        c.setFillColor(self.fg)
        c.setFont("Helvetica-Bold", self.font_size)
        c.drawString(10, 8, self.text)


def std_table(data, col_widths, header_bg=DARK, accent=ORANGE, font_size=9):
    """Build a styled table with alternating rows."""
    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), font_size),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING",    (0, 0), (-1, 0), 8),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), font_size - 0.5),
        ("TOPPADDING",  (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ROW_ALT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style_cmds))
    return t


# ════════════════════════════════════════════════════════════════════════════
# PRD
# ════════════════════════════════════════════════════════════════════════════

def build_prd(path="fraudwatch_PRD.pdf"):
    W, H = letter
    doc = SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.9*inch, bottomMargin=0.85*inch
    )
    usable = W - 1.7*inch

    # ── Styles ──
    title_s = style("Title", fontName="Helvetica-Bold", fontSize=26,
                    textColor=DARK, leading=30, spaceAfter=4, alignment=TA_LEFT)
    subtitle_s = style("Sub", fontName="Helvetica", fontSize=12,
                       textColor=ORANGE, spaceAfter=14, alignment=TA_LEFT)
    h1_s = style("H1", fontName="Helvetica-Bold", fontSize=13,
                 textColor=DARK, spaceBefore=18, spaceAfter=6)
    h2_s = style("H2", fontName="Helvetica-Bold", fontSize=10.5,
                 textColor=ORANGE, spaceBefore=10, spaceAfter=4)
    body_s = style("Body", fontName="Helvetica", fontSize=9.5, leading=14,
                   textColor=DARK, spaceAfter=6, alignment=TA_JUSTIFY)
    bullet_s = style("Bullet", fontName="Helvetica", fontSize=9.5, leading=13,
                     textColor=DARK, leftIndent=14, spaceAfter=3,
                     bulletIndent=4)
    meta_s = style("Meta", fontName="Helvetica", fontSize=8.5,
                   textColor=MUTED, spaceAfter=2)
    tag_s = style("Tag", fontName="Helvetica-Bold", fontSize=8,
                  textColor=WHITE, spaceAfter=0)
    given_s = style("Given", fontName="Helvetica-Oblique", fontSize=9.5,
                    leading=14, textColor=DARK, leftIndent=12, spaceAfter=3)

    story = []

    # ── 1. Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("FraudWatch", title_s))
    story.append(Paragraph("Product Requirements Document", subtitle_s))
    story.append(divider(ORANGE, 2))
    story.append(sp(4))

    meta_data = [
        ["Author", "Shrijani (Diya) Manna · Duke MEM '26"],
        ["GitHub", "github.com/diyaboop/fraud-watch"],
        ["Version", "1.0"],
        ["Status", "Draft"],
        ["Date", datetime.date.today().strftime("%B %d, %Y")],
        ["Audience", "Product managers, ML engineers, compliance teams, hiring reviewers"],
    ]
    meta_table = Table(meta_data, colWidths=[1.1*inch, usable - 1.1*inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), ORANGE),
        ("TEXTCOLOR", (1, 0), (1, -1), DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_table)
    story.append(sp(14))

    # ── 2. Overview ──────────────────────────────────────────────────────────
    story.append(ColorBlock("1 · Overview", bg=DARK))
    story.append(sp(6))
    story.append(Paragraph(
        "FraudWatch is an end-to-end fraud detection, bias audit, and model monitoring platform "
        "built on the IEEE-CIS Kaggle dataset (590,540 financial transactions, 3.5% fraud rate). "
        "It trains a Random Forest classifier, surfaces demographic disparities in false positive "
        "rates across card type, product category, and billing region, and tracks weekly "
        "performance drift — all served through a FastAPI backend with a live dashboard.",
        body_s))
    story.append(Paragraph(
        "The system is designed to demonstrate responsible AI development: it does not merely "
        "optimize for model accuracy but surfaces where automated decisions may systematically "
        "disadvantage certain transaction profiles, enabling human oversight and remediation.",
        body_s))

    # ── 3. Problem Statement ─────────────────────────────────────────────────
    story.append(sp(8))
    story.append(ColorBlock("2 · Problem Statement", bg=DARK))
    story.append(sp(6))

    story.append(Paragraph("Background", h2_s))
    story.append(Paragraph(
        "Fraud detection systems flag millions of transactions daily. False positives — legitimate "
        "transactions wrongly blocked — impose real customer harm: declined payments, frozen "
        "accounts, and loss of trust. When these errors are unevenly distributed across customer "
        "segments, the system creates a de-facto disparate impact even without any intent to "
        "discriminate.", body_s))

    story.append(Paragraph("Current State", h2_s))
    story.append(Paragraph(
        "Most deployed fraud models are evaluated purely on aggregate metrics (AUC, precision, "
        "recall). Demographic subgroup analysis is rarely part of the standard model card or "
        "release process. There is no live bias monitoring in production for most mid-market "
        "financial institutions.", body_s))

    story.append(Paragraph("Opportunity", h2_s))
    story.append(Paragraph(
        "FraudWatch demonstrates that bias auditing can be a first-class engineering concern — "
        "automated, reproducible, and queryable via API. The 45,375 false positives surfaced in "
        "a single test window, with a 65% disparity between credit and debit cardholders, "
        "represent a concrete, measurable harm that this system makes visible and actionable.",
        body_s))

    # ── 4. Goals & Success Metrics ───────────────────────────────────────────
    story.append(sp(8))
    story.append(ColorBlock("3 · Goals & Success Metrics", bg=DARK))
    story.append(sp(8))

    metrics_data = [
        ["Metric", "Baseline", "Target", "Status"],
        ["Fraud recall (test set)", "—", "≥ 0.85", "✓ 0.92"],
        ["FPR disparity surfaced (card type)", "—", "Quantified + significant", "✓ Z=66.6, p<0.001"],
        ["FPR disparity surfaced (product)", "—", "Quantified + significant", "✓ Z=109.1, p<0.001"],
        ["Weekly drift detection", "—", "0 undetected drift weeks", "✓ 0 / 12"],
        ["API response time (p95)", "—", "< 500ms", "In scope"],
        ["Audit reproducibility", "—", "Deterministic from seed", "✓ random_state=42"],
        ["Dashboard load time", "—", "< 3s on local network", "In scope"],
    ]
    story.append(std_table(metrics_data,
                           [1.8*inch, 1.2*inch, 2.1*inch, usable - 5.1*inch]))

    # ── 5. Users & User Stories ──────────────────────────────────────────────
    story.append(sp(8))
    story.append(ColorBlock("4 · Users & User Stories", bg=DARK))
    story.append(sp(6))

    personas = [
        ("ML Engineer / Data Scientist",
         "Trains and evaluates the fraud model. Needs reproducible pipelines, "
         "interpretable bias metrics, and drift alerts before production incidents."),
        ("Compliance / Risk Officer",
         "Responsible for regulatory adherence (ECOA, CFPB, SR 11-7). Needs "
         "documented evidence of bias monitoring and remediation workflows."),
        ("Product Manager / Analyst",
         "Owns the fraud product roadmap. Needs a clear dashboard summarizing "
         "model health, false positive burden, and actionable audit findings."),
    ]
    for name, desc in personas:
        story.append(Paragraph(f"<b>{name}</b>", h2_s))
        story.append(Paragraph(desc, body_s))

    story.append(sp(4))
    story.append(Paragraph("Epic 1 — Model Training & Evaluation", h2_s))
    us_e1 = [
        "As an <b>ML engineer</b>, I want a single-command pipeline that trains, evaluates, "
        "and saves the model so that onboarding new engineers takes under an hour.",
        "As an <b>ML engineer</b>, I want class-balanced training so that the 3.5% fraud rate "
        "does not cause the model to trivially predict 'not fraud' for everything.",
        "As a <b>PM</b>, I want a confusion matrix and classification report in the logs so "
        "that I can understand precision/recall trade-offs without writing code.",
    ]
    for u in us_e1:
        story.append(Paragraph(f"• {u}", bullet_s))

    story.append(sp(4))
    story.append(Paragraph("Epic 2 — Bias Audit", h2_s))
    us_e2 = [
        "As a <b>compliance officer</b>, I want false positive rates computed per card type "
        "so that I can assess ECOA disparate-impact risk before deployment.",
        "As a <b>compliance officer</b>, I want statistically significant disparity tests "
        "(Z-test, p-values) so that findings are defensible in a regulatory review.",
        "As an <b>ML engineer</b>, I want audit results persisted to SQLite so that I can "
        "query historical disparity trends without re-running the full pipeline.",
    ]
    for u in us_e2:
        story.append(Paragraph(f"• {u}", bullet_s))

    story.append(sp(4))
    story.append(Paragraph("Epic 3 — Monitoring & API", h2_s))
    us_e3 = [
        "As a <b>PM</b>, I want a live dashboard showing weekly recall and FPR so that I "
        "can brief leadership on model health without a data science handoff.",
        "As an <b>ML engineer</b>, I want drift alerts flagged automatically so that "
        "degradation is caught before it reaches end-users.",
        "As a <b>compliance officer</b>, I want a /predict endpoint so that individual "
        "transaction decisions can be audited in real time.",
    ]
    for u in us_e3:
        story.append(Paragraph(f"• {u}", bullet_s))

    # ── 6. Functional Requirements ───────────────────────────────────────────
    story.append(PageBreak())
    story.append(ColorBlock("5 · Functional Requirements", bg=DARK))
    story.append(sp(8))

    fr_data = [
        ["ID", "Priority", "Requirement"],
        ["FR-01", "Must",   "Train Random Forest on merged transaction + identity data"],
        ["FR-02", "Must",   "Apply class-balanced weights to handle 3.5% fraud minority class"],
        ["FR-03", "Must",   "Save model artifact and feature list to outputs/ via joblib"],
        ["FR-04", "Must",   "Compute FPR, FNR, and recall per card4, card6, ProductCD, addr2"],
        ["FR-05", "Must",   "Run proportions Z-test between key segment pairs"],
        ["FR-06", "Must",   "Persist audit_results and bias_summary to SQLite"],
        ["FR-07", "Must",   "Compute weekly precision, recall, FPR across 12 simulated weeks"],
        ["FR-08", "Must",   "Flag drift when recall < 0.85 or FPR > 0.55"],
        ["FR-09", "Must",   "Expose /metrics, /audit, /monitor, /predict via FastAPI"],
        ["FR-10", "Must",   "Serve single-page dashboard at GET /"],
        ["FR-11", "Should", "Return bias charts as base64 PNG embedded in JSON"],
        ["FR-12", "Should", "Containerize full app in a single Dockerfile"],
        ["FR-13", "Could",  "Accept custom threshold parameter in /predict"],
        ["FR-14", "Won't",  "Real-time streaming from live transaction feeds (v2 scope)"],
    ]
    story.append(std_table(fr_data, [0.7*inch, 0.85*inch, usable - 1.55*inch]))

    # ── 7. Non-Functional Requirements ──────────────────────────────────────
    story.append(sp(10))
    story.append(ColorBlock("6 · Non-Functional Requirements", bg=DARK))
    story.append(sp(6))

    nfr = [
        ("Performance", "API p95 latency < 500ms; dashboard first paint < 3s on local network."),
        ("Reproducibility", "All training, audit, and monitor runs produce identical outputs with random_state=42."),
        ("Portability", "Docker image runs on any Linux/macOS host with Docker installed; no GPU required."),
        ("Security", "No PII in API responses; model artifact not exposed directly via HTTP."),
        ("Observability", "Drift alerts surfaced in /monitor JSON response and in uvicorn server logs."),
        ("Maintainability", "ML pipeline (train.py, audit.py, monitor.py) untouched by API layer; clean separation of concerns."),
    ]
    for name, desc in nfr:
        story.append(Paragraph(f"<b>{name}:</b>  {desc}", body_s))

    # ── 8. System Design ─────────────────────────────────────────────────────
    story.append(sp(8))
    story.append(ColorBlock("7 · System Design", bg=DARK))
    story.append(sp(6))

    story.append(Paragraph("Architecture", h2_s))
    arch_data = [
        ["Layer", "Component", "Technology"],
        ["Data",       "IEEE-CIS CSVs (590K rows)",         "Pandas, CSV"],
        ["Training",   "Random Forest (100 trees, depth=10)","Scikit-learn"],
        ["Audit",      "FPR disparity + Z-tests",            "SciPy / statsmodels"],
        ["Monitoring", "12-week drift simulation",           "Pandas, qcut"],
        ["Persistence","Model + metrics + audit results",    "joblib, SQLite"],
        ["API",        "REST endpoints (/metrics /audit …)", "FastAPI, Uvicorn"],
        ["Frontend",   "Single-page dashboard",              "Vanilla JS, HTML/CSS"],
        ["Deployment", "Containerized app",                  "Docker"],
    ]
    story.append(std_table(arch_data, [1.1*inch, 2.3*inch, usable - 3.4*inch]))

    story.append(sp(8))
    story.append(Paragraph("API Endpoints", h2_s))
    api_data = [
        ["Method", "Path",     "Description",                          "Auth"],
        ["GET",  "/metrics",  "Aggregate KPIs (recall, precision, FPR, totals)", "None"],
        ["GET",  "/audit",    "Bias audit data + base64 chart PNG",    "None"],
        ["GET",  "/monitor",  "Weekly metrics + drift flags + chart",  "None"],
        ["POST", "/predict",  "Score single transaction (30 features)","None"],
        ["GET",  "/",        "Serve SPA dashboard (index.html)",       "None"],
    ]
    story.append(std_table(api_data,
                           [0.65*inch, 0.9*inch, 2.9*inch, usable - 4.45*inch]))

    story.append(sp(8))
    story.append(Paragraph("Data Flow", h2_s))
    story.append(Paragraph(
        "1. train.py reads CSVs → feature engineering → model fit → saves fraud_model.pkl + features.pkl  "
        "2. audit.py loads model → runs inference on test split → computes per-segment FPR → writes to fraudwatch.db  "
        "3. monitor.py loads model → simulates 12 weekly batches → computes drift metrics → writes weekly_metrics to fraudwatch.db  "
        "4. api.py reads fraudwatch.db at request time → generates charts on-the-fly → returns JSON",
        body_s))

    # ── 9. Acceptance Criteria ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(ColorBlock("8 · Acceptance Criteria", bg=DARK))
    story.append(sp(6))

    ac = [
        ("AC-01 · Training pipeline",
         "Given the IEEE-CIS CSV files exist in data/,\n"
         "When train.py is executed,\n"
         "Then outputs/fraud_model.pkl and outputs/features.pkl are created, "
         "model recall on the test set exceeds 0.85, and the run completes without error."),
        ("AC-02 · Bias audit persistence",
         "Given fraud_model.pkl exists,\n"
         "When audit.py is executed,\n"
         "Then fraudwatch.db contains tables audit_results and bias_summary with non-empty rows, "
         "and the credit vs debit Z-stat is ≥ 60."),
        ("AC-03 · Drift monitoring",
         "Given fraud_model.pkl exists,\n"
         "When monitor.py is executed,\n"
         "Then fraudwatch.db contains 12 rows in weekly_metrics and drift_flag is 0 for all weeks "
         "given stable simulated data."),
        ("AC-04 · API liveness",
         "Given the FastAPI server is running on port 8000,\n"
         "When GET /metrics is called,\n"
         "Then the response status is 200 and contains avg_recall, avg_fpr, total_transactions."),
        ("AC-05 · Predict endpoint",
         "Given the server is running,\n"
         "When POST /predict is called with a valid 30-feature JSON payload,\n"
         "Then the response includes fraud_probability (0–1), flagged (bool), and risk_level (LOW/MEDIUM/HIGH)."),
        ("AC-06 · Docker build",
         "Given Docker is installed,\n"
         "When 'docker build -t fraudwatch .' is run,\n"
         "Then the image builds without error and 'docker run -p 8000:8000 fraudwatch' serves all endpoints."),
    ]
    for title, text in ac:
        story.append(KeepTogether([
            Paragraph(f"<b>{title}</b>", h2_s),
            Paragraph(text.replace("\n", "<br/>"), given_s),
            sp(4),
        ]))

    # ── 10. Risks & Open Questions ───────────────────────────────────────────
    story.append(sp(4))
    story.append(ColorBlock("9 · Risks & Open Questions", bg=DARK))
    story.append(sp(8))

    risk_data = [
        ["Risk", "Level", "Mitigation"],
        ["Simulated weeks from a single test split — not true temporal drift",
         "Medium", "Label as 'simulated'; v2 to use rolling time windows from TransactionDT"],
        ["High FPR (39.8% avg) due to 0.30 probability threshold",
         "High",   "Threshold is tunable; surfaced explicitly in dashboard and docs"],
        ["card6 and ProductCD encoding (int codes) may not generalize",
         "Medium", "Encodings are deterministic from same data split; document in model card"],
        ["No authentication on API endpoints",
         "Low",    "Localhost / internal use only; add API key middleware for production"],
        ["Large CSV data files not included in Docker image",
         "Medium", "Pre-computed outputs/ shipped; retrain requires data re-mount"],
        ["Non-US FPR finding (77–85%) based on addr2 codes — proxy, not ground truth",
         "Medium", "Document as proxy signal; pair with demographic validation in production"],
    ]
    story.append(std_table(risk_data, [3.0*inch, 0.75*inch, usable - 3.75*inch]))

    story.append(sp(10))
    story.append(Paragraph("Open Questions", h2_s))
    oq = [
        "Should the 0.30 fraud probability threshold be configurable per product line?",
        "What is the right FPR disparity threshold to trigger a mandatory model review?",
        "How should the system handle missing identity data (left-join nulls → -999 fill)?",
        "Is addr2 a reliable proxy for billing region, or should IP geolocation be used?",
    ]
    for q in oq:
        story.append(Paragraph(f"• {q}", bullet_s))

    # ── 11. Out of Scope & Regulatory Mapping ────────────────────────────────
    story.append(sp(8))
    story.append(ColorBlock("10 · Out of Scope & Regulatory Mapping", bg=DARK))
    story.append(sp(6))

    story.append(Paragraph("Out of Scope (v1)", h2_s))
    oos = [
        "Real-time transaction streaming or Kafka integration",
        "Model retraining automation (MLflow, Airflow)",
        "Multi-model A/B testing or shadow deployment",
        "End-user authentication and role-based access control",
        "Production-grade alerting (PagerDuty, Slack webhooks)",
    ]
    for o in oos:
        story.append(Paragraph(f"• {o}", bullet_s))

    story.append(sp(6))
    story.append(Paragraph("Regulatory Mapping", h2_s))
    reg_data = [
        ["Regulation", "Applicability", "How FraudWatch Addresses It"],
        ["ECOA (Equal Credit Opportunity Act)",
         "High",
         "FPR disparity audit across card type surfaces potential disparate impact in credit-adjacent decisions"],
        ["EU AI Act (High-Risk AI, Annex III)",
         "High",
         "Bias audit, documentation, and human oversight mechanisms directly map to Art. 9–13 requirements"],
        ["CFPB Unfair, Deceptive, or Abusive Acts",
         "Medium",
         "False positive burden on customers constitutes potential harm; quantification enables remediation"],
        ["SR 11-7 (Model Risk Management)",
         "High",
         "Weekly performance monitoring, drift detection, and documented acceptance criteria fulfill SR 11-7 validation requirements"],
    ]
    story.append(std_table(reg_data,
                           [1.8*inch, 0.7*inch, usable - 2.5*inch],
                           header_bg=GREEN))

    story.append(sp(14))
    story.append(divider(MUTED, 0.5))
    story.append(Paragraph(
        "FraudWatch · Shrijani (Diya) Manna · Duke MEM '26 · github.com/diyaboop/fraud-watch",
        style("Footer", fontName="Helvetica", fontSize=7.5, textColor=MUTED,
              alignment=TA_CENTER)))

    doc.build(story)
    print(f"✓  PRD saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# CASE STUDY  (1-pager, A4 landscape feels cramped — use letter portrait tight)
# ════════════════════════════════════════════════════════════════════════════

def build_case_study(path="fraudwatch_case_study.pdf"):
    W, H = letter
    LM = RM = 0.7 * inch
    TM = BM = 0.65 * inch
    usable = W - LM - RM

    doc = SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM, bottomMargin=BM
    )

    # ── Styles ──
    proj_s   = style("ProjName", fontName="Helvetica-Bold", fontSize=28,
                     textColor=DARK, leading=32, spaceAfter=2)
    tag_s    = style("Tagline",  fontName="Helvetica-Oblique", fontSize=12,
                     textColor=ORANGE, spaceAfter=10)
    h1_s     = style("CSH1", fontName="Helvetica-Bold", fontSize=9,
                     textColor=WHITE, spaceAfter=0)
    body_s   = style("CSBody", fontName="Helvetica", fontSize=8.5, leading=12.5,
                     textColor=DARK, spaceAfter=4)
    bullet_s = style("CSBullet", fontName="Helvetica", fontSize=8.5, leading=12,
                     textColor=DARK, leftIndent=10, spaceAfter=2)
    foot_s   = style("CSFoot", fontName="Helvetica", fontSize=7.5,
                     textColor=MUTED, alignment=TA_CENTER)

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("FraudWatch", proj_s))
    story.append(Paragraph(
        "Bias-Audited Fraud Detection · IEEE-CIS · Random Forest · FastAPI · Docker",
        tag_s))
    story.append(divider(ORANGE, 2, spaceBefore=0, spaceAfter=8))

    # ── Two-column top section: Problem | Architecture ───────────────────────
    prob_text = (
        "<b>Problem:</b> Fraud detection models flag millions of transactions daily. "
        "When false positives concentrate on specific customer profiles — credit card "
        "holders, certain product categories, non-US billing addresses — the system "
        "creates measurable disparate impact without any discriminatory intent.<br/><br/>"
        "FraudWatch makes this invisible harm visible: it trains a classifier on 590,540 "
        "real transactions, quantifies where errors are unevenly distributed, and monitors "
        "performance weekly — all queryable through a live API."
    )
    arch_text = (
        "<b>Stack:</b> Python · Scikit-learn · Pandas · SciPy<br/>"
        "SQLite · FastAPI · Uvicorn · Docker<br/><br/>"
        "<b>Data:</b> IEEE-CIS Kaggle dataset<br/>"
        "590,540 transactions · 3.5% fraud rate<br/>"
        "Train/test split 80/20 · stratified<br/><br/>"
        "<b>Model:</b> Random Forest · 100 trees · depth=10<br/>"
        "class_weight='balanced' · random_state=42"
    )
    top_table = Table(
        [[Paragraph(prob_text, body_s), Paragraph(arch_text, body_s)]],
        colWidths=[usable * 0.58, usable * 0.42]
    )
    top_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    story.append(top_table)
    story.append(sp(10))

    # ── KPI stat cards ───────────────────────────────────────────────────────
    def stat_cell(value, label, color=DARK):
        return [
            Paragraph(f'<font color="{color.hexval()}" size="22"><b>{value}</b></font>',
                      style(f"KV{value}", fontName="Helvetica-Bold", fontSize=22,
                            textColor=color, alignment=TA_CENTER, leading=26)),
            Paragraph(label, style(f"KL{value}", fontName="Helvetica", fontSize=7.5,
                                   textColor=MUTED, alignment=TA_CENTER, leading=10)),
        ]

    def kpi_cell(value, label, color):
        inner = Table(
            [stat_cell(value, label, color)],
            colWidths=[(usable / 4) - 6]
        )
        inner.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        return inner

    kpi_row = [[
        kpi_cell("92.4%",   "Avg Recall\n(12 weeks)", GREEN),
        kpi_cell("45,375",  "False Positives\nidentified", ORANGE),
        kpi_cell("2.7×",    "FPR Disparity\nProduct Cat 0 vs 4", PURPLE),
        kpi_cell("0 / 12",  "Drift Weeks\ndetected", DARK),
    ]]
    kpi_table = Table(kpi_row, colWidths=[(usable / 4)] * 4)
    kpi_table.setStyle(TableStyle([
        ("BOX", (0, 0), (0, 0), 1, MID),
        ("BOX", (1, 0), (1, 0), 1, MID),
        ("BOX", (2, 0), (2, 0), 1, MID),
        ("BOX", (3, 0), (3, 0), 1, MID),
        ("BACKGROUND", (0, 0), (-1, -1), BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(kpi_table)
    story.append(sp(10))

    # ── Bias Audit Findings ──────────────────────────────────────────────────
    story.append(ColorBlock("Bias Audit Findings", bg=DARK, height=22, font_size=10))
    story.append(sp(6))
    bias_data = [
        ["Segment", "Group A", "FPR A", "Group B", "FPR B", "Disparity", "Z-stat", "Significant"],
        ["Card Type (card6)", "Credit", "56.8%", "Debit", "34.3%", "+22.5 pp  (1.66×)", "Z=66.6", "p < 0.001"],
        ["Product Category", "Cat 0",  "82.3%", "Cat 4", "30.8%", "+51.5 pp  (2.67×)", "Z=109.1","p < 0.001"],
        ["Billing Region",   "Non-US", "77–85%","US",    "34.8%", "+42–50 pp (2.2×)",  "—",      "Notable"],
    ]
    story.append(std_table(bias_data,
                           [1.3*inch, 0.7*inch, 0.55*inch, 0.6*inch, 0.55*inch,
                            1.35*inch, 0.8*inch, usable - 5.85*inch],
                           font_size=8))

    story.append(sp(10))

    # ── Three-column bottom: User Stories | Business Impact | Regulatory ─────
    us_content = [
        ColorBlock("User Stories", bg=GREEN, height=20, font_size=9),
        sp(4),
        Paragraph("• As a <b>compliance officer</b>, I want FPR computed per card type so I can assess ECOA disparate-impact risk before deployment.", bullet_s),
        Paragraph("• As an <b>ML engineer</b>, I want drift alerts flagged automatically so degradation is caught before it reaches end-users.", bullet_s),
        Paragraph("• As a <b>PM</b>, I want a live dashboard showing weekly recall so I can brief leadership without a data science handoff.", bullet_s),
    ]

    impact_content = [
        ColorBlock("Business Impact", bg=PURPLE, height=20, font_size=9),
        sp(4),
        Paragraph("• <b>45,375 false positives</b> surfaced in a single test window — each represents a wrongly blocked customer transaction.", bullet_s),
        Paragraph("• Credit cardholders are <b>65% more likely</b> to be wrongly flagged than debit holders — a regulatory liability.", bullet_s),
        Paragraph("• <b>Zero drift</b> across 12 weeks confirms model stability and enables confident production deployment.", bullet_s),
        Paragraph("• Full pipeline reproducible with one command — <b>reducing audit prep time</b> from days to minutes.", bullet_s),
    ]

    reg_content = [
        ColorBlock("Regulatory Context", bg=ORANGE, height=20, font_size=9),
        sp(4),
        Paragraph("<b>ECOA</b> — Disparate impact quantified across protected-adjacent attributes (card type, region).", bullet_s),
        Paragraph("<b>EU AI Act</b> — Bias audit, human oversight, and monitoring docs map to Art. 9–13.", bullet_s),
        Paragraph("<b>CFPB UDAAP</b> — False positive burden constitutes potential consumer harm; now measurable.", bullet_s),
        Paragraph("<b>SR 11-7</b> — Weekly drift monitoring and acceptance criteria satisfy model validation requirements.", bullet_s),
    ]

    col_w = (usable - 12) / 3

    def cell_from_list(items):
        t = Table([[item] for item in items], colWidths=[col_w])
        t.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    bottom = Table(
        [[cell_from_list(us_content), cell_from_list(impact_content), cell_from_list(reg_content)]],
        colWidths=[col_w, col_w, col_w],
        hAlign="LEFT"
    )
    bottom.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LINEAFTER", (0, 0), (1, 0), 0.5, MID),
    ]))
    story.append(bottom)

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(sp(8))
    story.append(divider(MUTED, 0.5, spaceBefore=4, spaceAfter=4))
    story.append(Paragraph(
        "Shrijani (Diya) Manna · Duke Master of Engineering Management '26 · "
        "github.com/diyaboop/fraud-watch · "
        + datetime.date.today().strftime("%B %Y"),
        foot_s))

    doc.build(story)
    print(f"✓  Case Study saved → {path}")


if __name__ == "__main__":
    import os
    os.chdir("/Users/diyamanna/fraudwatch")
    build_prd("fraudwatch_PRD.pdf")
    build_case_study("fraudwatch_case_study.pdf")

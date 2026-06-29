import json
import re
import os
import streamlit as st
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSEN_JSON = os.path.join(BASE_DIR, "dosen_sinta.json")
KURIKULUM_PDF = os.path.join(BASE_DIR, "Kurikulum Sadat.pdf")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── CSS ─────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; box-sizing: border-box; }

/* ── page ── */
.stApp { background: linear-gradient(160deg,#EEF6FF 0%,#DCEEFB 50%,#F0F8FF 100%); min-height:100vh; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding-top:0.6rem; padding-bottom:3rem; max-width:1100px; }

/* ── keyframes ── */
@keyframes floatBlob {
    0%,100% { transform:translate(0,0) scale(1); }
    50%      { transform:translate(15px,-15px) scale(1.1); }
}
@keyframes floatAround {
    0%,100% { transform:translate(0,0) rotate(0deg) scale(1); }
    33%      { transform:translate(28px,-28px) rotate(120deg) scale(1.08); }
    66%      { transform:translate(-20px,18px) rotate(240deg) scale(0.94); }
}
@keyframes driftA {
    0%,100% { transform:translate(0,0) scale(1); }
    50%      { transform:translate(35px,-22px) scale(1.07); }
}
@keyframes driftB {
    0%,100% { transform:translate(0,0) scale(1); }
    50%      { transform:translate(-28px,25px) scale(0.94); }
}
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(22px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes cardIn {
    from { opacity:0; transform:translateY(14px) scale(0.97); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
@keyframes shimmer {
    0%   { background-position:-200% center; }
    100% { background-position: 200% center; }
}
@keyframes btnPulse {
    0%,100% { box-shadow:0 4px 20px rgba(25,118,210,0.32); }
    50%      { box-shadow:0 6px 32px rgba(25,118,210,0.58); }
}

/* ── main page ambient blobs ── */
.mb1,.mb2,.mb3 { position:fixed; border-radius:50%; pointer-events:none; z-index:0; }
.mb1 {
    width:520px; height:520px; top:-160px; right:-140px;
    background:radial-gradient(circle,rgba(100,181,246,0.13) 0%,transparent 65%);
    animation:driftA 16s ease-in-out infinite;
}
.mb2 {
    width:400px; height:400px; bottom:-110px; left:-110px;
    background:radial-gradient(circle,rgba(144,202,249,0.15) 0%,transparent 65%);
    animation:driftB 20s ease-in-out infinite;
}
.mb3 {
    width:280px; height:280px; top:42%; right:4%;
    background:radial-gradient(circle,rgba(33,150,243,0.09) 0%,transparent 65%);
    animation:driftA 11s ease-in-out infinite; animation-delay:-5s;
}

/* ── entrance animations ── */
.hero        { animation:fadeInUp 0.45s ease both; }
.input-wrap  { animation:fadeInUp 0.5s ease both 0.06s; }
.status-banner { animation:fadeInUp 0.4s ease both; }
.kw-section  { animation:fadeInUp 0.45s ease both 0.08s; }
.mk-card     { animation:cardIn 0.4s ease both; }
.llm-section { animation:cardIn 0.4s ease both; }
.feat-card   { animation:cardIn 0.5s ease both; }
.feat-card:nth-child(1){ animation-delay:0.05s; }
.feat-card:nth-child(2){ animation-delay:0.13s; }
.feat-card:nth-child(3){ animation-delay:0.21s; }
.feat-card:nth-child(4){ animation-delay:0.29s; }

/* ── hero ── */
.hero {
    background:linear-gradient(135deg,rgba(144,202,249,0.38) 0%,rgba(227,242,253,0.75) 100%);
    border:1.5px solid rgba(33,150,243,0.22);
    border-radius:22px; padding:1.6rem 2rem 1.5rem;
    margin-bottom:1rem; position:relative; overflow:hidden;
    box-shadow:0 8px 40px rgba(33,150,243,0.1),inset 0 1px 0 rgba(255,255,255,0.8);
    animation:fadeInUp 0.45s ease both;
}
.hero::before {
    content:''; position:absolute; top:-50px; right:-50px;
    width:180px; height:180px;
    background:radial-gradient(circle,rgba(100,181,246,0.25) 0%,transparent 70%);
    border-radius:50%; animation:floatBlob 6s ease-in-out infinite;
}
.hero::after {
    content:''; position:absolute; bottom:-35px; left:-35px;
    width:140px; height:140px;
    background:radial-gradient(circle,rgba(144,202,249,0.28) 0%,transparent 70%);
    border-radius:50%; animation:floatBlob 8s ease-in-out infinite reverse;
}

/* hero top bar */
.hero-topbar {
    display:flex; align-items:center; justify-content:space-between;
    flex-wrap:wrap; gap:0.8rem; margin-bottom:1.4rem;
}
.hero-brand { display:flex; align-items:center; gap:0.9rem; }
.hero-title {
    font-size:1.9rem; font-weight:800; line-height:1;
    background:linear-gradient(135deg,#0D47A1,#1976D2,#42A5F5);
    background-size:200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:shimmer 6s linear infinite;
    margin:0 0 0.15rem;
}
.hero-sub { color:rgba(21,101,192,0.55); font-size:0.78rem; margin:0; font-weight:500; }
.hero-pills { display:flex; gap:0.5rem; flex-wrap:wrap; }
.hero-pill {
    background:rgba(255,255,255,0.7); border:1px solid rgba(33,150,243,0.22);
    border-radius:999px; padding:0.28rem 0.85rem;
    font-size:0.7rem; font-weight:700; color:#1565C0;
    display:flex; align-items:center; gap:0.4rem;
}
.hero-pill-num { font-size:0.95rem; font-weight:800; color:#1976D2; }

/* how it works */
.how-wrap {
    display:flex; align-items:stretch; gap:0; position:relative;
}
.how-step {
    flex:1; background:rgba(255,255,255,0.62);
    border:1px solid rgba(33,150,243,0.18); border-radius:14px;
    padding:1rem 1.1rem; position:relative;
    transition:all 0.25s ease;
}
.how-step:hover { background:rgba(255,255,255,0.85); transform:translateY(-2px); box-shadow:0 6px 20px rgba(25,118,210,0.1); }
.how-num {
    font-size:0.58rem; font-weight:800; letter-spacing:0.12em;
    color:rgba(25,118,210,0.4); text-transform:uppercase; margin-bottom:0.5rem;
}
.how-icon {
    width:32px; height:32px; border-radius:10px;
    background:linear-gradient(135deg,rgba(144,202,249,0.5),rgba(187,222,251,0.6));
    border:1px solid rgba(33,150,243,0.2);
    display:flex; align-items:center; justify-content:center;
    font-size:0.9rem; margin-bottom:0.55rem;
}
.how-title { font-size:0.85rem; font-weight:700; color:#0D47A1; margin-bottom:0.3rem; }
.how-desc  { font-size:0.73rem; color:rgba(13,59,110,0.55); line-height:1.55; }
.how-arrow {
    display:flex; align-items:center; padding:0 0.5rem;
    color:rgba(33,150,243,0.4); font-size:1.1rem; flex-shrink:0;
    font-weight:300;
}

/* ── input section ── */
.input-wrap {
    background:rgba(255,255,255,0.78); border:1.5px solid rgba(33,150,243,0.2);
    border-radius:18px; padding:1.4rem 1.6rem; margin-bottom:1rem;
    box-shadow:0 4px 24px rgba(33,150,243,0.07); backdrop-filter:blur(10px);
    transition:box-shadow 0.3s ease;
}
.input-wrap:focus-within { box-shadow:0 6px 32px rgba(25,118,210,0.14); }
.input-label { color:#1565C0; font-size:0.8rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; margin-bottom:0.5rem; }
.input-hint { color:rgba(21,101,192,0.5); font-size:0.78rem; margin-top:0.4rem; }

textarea {
    background:rgba(240,248,255,0.9) !important;
    border:1.5px solid rgba(33,150,243,0.28) !important;
    border-radius:14px !important; color:#0D3B6E !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:0.95rem !important; line-height:1.6 !important;
    transition:all 0.25s ease !important; resize:vertical !important;
}
textarea:focus {
    border-color:#1976D2 !important;
    box-shadow:0 0 0 3px rgba(25,118,210,0.12) !important;
    background:white !important;
}
textarea::placeholder { color:rgba(21,101,192,0.35) !important; }

/* ── example pills ── */
.example-wrap { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.6rem; }
.example-pill {
    background:rgba(144,202,249,0.22); border:1px solid rgba(33,150,243,0.25);
    color:#1565C0; font-size:0.73rem; font-weight:600;
    padding:0.25rem 0.8rem; border-radius:999px; cursor:pointer;
    transition:all 0.22s ease; display:inline-block;
}
.example-pill:hover { background:rgba(100,181,246,0.35); border-color:rgba(25,118,210,0.45); transform:scale(1.05) translateY(-1px); }

/* ── primary button ── */
.stButton>button[kind="primary"] {
    background:linear-gradient(135deg,#1565C0,#1976D2,#42A5F5) !important;
    background-size:200% auto !important;
    border:none !important; border-radius:14px !important;
    color:white !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:1rem !important; font-weight:700 !important;
    padding:0.8rem 2rem !important; letter-spacing:0.03em !important;
    transition:all 0.3s ease !important;
    animation:btnPulse 2.8s ease-in-out infinite !important;
}
.stButton>button[kind="primary"]:hover {
    transform:translateY(-3px) scale(1.02) !important;
    box-shadow:0 10px 32px rgba(25,118,210,0.5) !important;
    animation:none !important;
}
.stButton>button[kind="primary"]:active { transform:translateY(0) scale(0.98) !important; }

/* ── secondary button (back) ── */
.stButton>button[kind="secondary"] {
    background:rgba(227,242,253,0.8) !important;
    border:1.5px solid rgba(33,150,243,0.3) !important;
    border-radius:10px !important; color:#1565C0 !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:0.82rem !important; font-weight:600 !important;
    padding:0.4rem 1rem !important;
    transition:all 0.22s ease !important;
}
.stButton>button[kind="secondary"]:hover {
    background:rgba(187,222,251,0.6) !important;
    border-color:rgba(25,118,210,0.5) !important;
    transform:translateX(-3px) !important;
}

/* ── step tracker ── */
.steps-wrap { display:flex; align-items:center; justify-content:center; gap:0; margin:0.8rem 0 1rem; flex-wrap:wrap; }
.step { display:flex; align-items:center; gap:0.5rem; padding:0.5rem 1rem; border-radius:999px; font-size:0.8rem; font-weight:600; }
.step-done    { background:rgba(25,118,210,0.12); color:#1565C0; }
.step-active  { background:linear-gradient(135deg,#1565C0,#42A5F5); color:white; box-shadow:0 3px 12px rgba(25,118,210,0.35); }
.step-pending { background:rgba(255,255,255,0.5); color:rgba(21,101,192,0.4); border:1px solid rgba(33,150,243,0.15); }
.step-line      { width:40px; height:2px; background:rgba(33,150,243,0.2); flex-shrink:0; }
.step-line-done { background:rgba(25,118,210,0.45); }

/* ── status banner ── */
.status-banner {
    border-radius:18px; padding:1.2rem 1.6rem;
    display:flex; align-items:center; gap:1rem;
    margin:1rem 0 1.5rem; box-shadow:0 4px 20px rgba(0,0,0,0.05);
    transition:transform 0.25s ease, box-shadow 0.25s ease;
}
.status-banner:hover { transform:translateY(-2px); box-shadow:0 8px 28px rgba(0,0,0,0.08); }
.sb-green  { background:linear-gradient(135deg,#d1fae5,#ecfdf5); border:1.5px solid #6ee7b7; }
.sb-blue   { background:linear-gradient(135deg,#dbeafe,#eff6ff); border:1.5px solid #93c5fd; }
.sb-orange { background:linear-gradient(135deg,#fef3c7,#fffbeb); border:1.5px solid #fcd34d; }
.sb-red    { background:linear-gradient(135deg,#fee2e2,#fff5f5); border:1.5px solid #fca5a5; }
.sb-icon { font-size:1.6rem; font-weight:900; flex-shrink:0; line-height:1; }
.sb-title { font-weight:800; font-size:1.05rem; }
.sb-green  .sb-title { color:#065f46; }
.sb-blue   .sb-title { color:#1e40af; }
.sb-orange .sb-title { color:#78350f; }
.sb-red    .sb-title { color:#7f1d1d; }
.sb-desc { font-size:0.82rem; margin-top:0.2rem; opacity:0.75; }

/* ── donut score ── */
.donut-wrap { display:flex; flex-direction:column; align-items:center; gap:0.3rem; }
.donut-ring { position:relative; width:90px; height:90px; transition:transform 0.3s ease; }
.donut-ring:hover { transform:scale(1.1); }
.donut-ring svg { transform:rotate(-90deg); }
.donut-center { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center; }
.donut-num { font-size:1.25rem; font-weight:800; color:#1565C0; display:block; line-height:1; }
.donut-sub { font-size:0.58rem; color:rgba(21,101,192,0.55); font-weight:600; text-transform:uppercase; }
.donut-label { font-size:0.72rem; font-weight:600; color:rgba(21,101,192,0.6); }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,0.62) !important; border-radius:14px !important;
    padding:0.3rem !important; border:1px solid rgba(33,150,243,0.15) !important; gap:0.3rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius:10px !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:0.85rem !important; font-weight:600 !important;
    color:rgba(21,101,192,0.6) !important; padding:0.5rem 1.2rem !important;
    border:none !important; transition:all 0.22s ease !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#1565C0,#42A5F5) !important;
    color:white !important; box-shadow:0 3px 12px rgba(25,118,210,0.28) !important;
}
.stTabs [data-baseweb="tab-border"]    { display:none !important; }
.stTabs [data-baseweb="tab-highlight"] { display:none !important; }
.stTabs [data-baseweb="tab-panel"]     { padding-top:1.5rem !important; }

/* ── section heading ── */
.sec-head {
    font-size:1rem; font-weight:700; color:#1565C0;
    display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem;
}
.sec-head::after { content:''; flex:1; height:1.5px; background:linear-gradient(90deg,rgba(25,118,210,0.3),transparent); }

/* ── kurikulum card ── */
.mk-card {
    background:rgba(255,255,255,0.72); border:1.5px solid rgba(33,150,243,0.18);
    border-radius:18px; padding:1.4rem 1.6rem; margin-bottom:1rem;
    box-shadow:0 2px 12px rgba(33,150,243,0.06); transition:all 0.28s ease;
}
.mk-card:hover { border-color:rgba(25,118,210,0.4); transform:translateY(-3px) scale(1.01); box-shadow:0 8px 28px rgba(25,118,210,0.12); }
.mk-header { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin-bottom:0.8rem; }
.mk-name { font-weight:700; color:#0D47A1; font-size:0.95rem; }
.mk-score-pill {
    background:rgba(144,202,249,0.3); border:1px solid rgba(25,118,210,0.25);
    color:#1565C0; font-size:0.7rem; font-weight:700;
    padding:0.2rem 0.7rem; border-radius:999px; flex-shrink:0;
}
.mk-bar-track { background:rgba(144,202,249,0.25); border-radius:999px; height:6px; overflow:hidden; margin:0.4rem 0 0.8rem; }
.mk-bar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#90CAF9,#1565C0); }
.mk-body { color:rgba(13,59,110,0.62); font-size:0.8rem; line-height:1.65; }

/* ── dosen cards ── */
.dosen-utama {
    background:linear-gradient(135deg,rgba(144,202,249,0.38),rgba(227,242,253,0.7));
    border:2px solid rgba(25,118,210,0.28); border-radius:22px; padding:1.6rem;
    margin-bottom:1.2rem; box-shadow:0 6px 28px rgba(25,118,210,0.1);
    position:relative; transition:all 0.28s ease;
}
.dosen-utama:hover { transform:translateY(-4px); box-shadow:0 14px 40px rgba(25,118,210,0.18); }
.dosen-alt {
    background:rgba(255,255,255,0.65); border:1.5px solid rgba(33,150,243,0.18);
    border-radius:18px; padding:1.4rem; margin-bottom:0.9rem;
    box-shadow:0 2px 10px rgba(33,150,243,0.05); transition:all 0.28s ease;
}
.dosen-alt:hover { border-color:rgba(25,118,210,0.38); transform:translateY(-3px) scale(1.005); box-shadow:0 8px 22px rgba(25,118,210,0.12); }

.dosen-top { display:flex; align-items:center; gap:1rem; margin-bottom:0.9rem; }
.avatar {
    width:50px; height:50px; border-radius:50%;
    background:linear-gradient(135deg,#1565C0,#42A5F5);
    display:flex; align-items:center; justify-content:center;
    font-weight:800; font-size:1.1rem; color:white; flex-shrink:0;
    box-shadow:0 3px 12px rgba(25,118,210,0.35);
    transition:transform 0.25s ease;
}
.dosen-utama:hover .avatar { transform:scale(1.1) rotate(-5deg); }
.avatar-sm {
    width:40px; height:40px; border-radius:50%;
    background:linear-gradient(135deg,#42A5F5,#90CAF9);
    display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:0.9rem; color:white; flex-shrink:0;
    transition:transform 0.25s ease;
}
.dosen-alt:hover .avatar-sm { transform:scale(1.12); }
.dosen-name { font-weight:800; color:#0D47A1; font-size:1rem; margin-bottom:0.2rem; }
.dosen-role { font-size:0.73rem; color:rgba(21,101,192,0.5); font-weight:600; }

.tag-wrap { display:flex; flex-wrap:wrap; gap:0.35rem; margin:0.6rem 0; }
.tag {
    background:rgba(144,202,249,0.28); border:1px solid rgba(33,150,243,0.28);
    color:#1565C0; font-size:0.7rem; font-weight:600;
    padding:0.18rem 0.65rem; border-radius:999px;
    transition:all 0.22s ease;
}
.tag:hover { background:rgba(100,181,246,0.45); transform:scale(1.06) translateY(-1px); }

.score-row { display:flex; gap:1.5rem; margin-top:0.8rem; flex-wrap:wrap; }
.score-item { text-align:center; transition:transform 0.22s ease; cursor:default; }
.score-item:hover { transform:translateY(-4px); }
.score-num { font-size:1.1rem; font-weight:800; color:#1565C0; display:block; }
.score-lbl { font-size:0.65rem; color:rgba(21,101,192,0.5); font-weight:600; text-transform:uppercase; }
.score-bar-mini { width:60px; height:4px; background:rgba(144,202,249,0.3); border-radius:999px; overflow:hidden; margin:0.3rem auto 0; }
.score-bar-mini-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#90CAF9,#1565C0); }

.utama-badge {
    position:absolute; top:1rem; right:1rem;
    background:rgba(219,234,254,0.85); border:1px solid rgba(147,197,253,0.6);
    color:#1e40af; font-size:0.65rem; font-weight:700;
    letter-spacing:0.08em; padding:0.2rem 0.6rem; border-radius:999px;
}

/* ── keyword chips ── */
.kw-section { margin:0.5rem 0 1.5rem; }
.kw-label { font-size:0.72rem; font-weight:700; color:rgba(21,101,192,0.5); letter-spacing:0.07em; text-transform:uppercase; margin-bottom:0.4rem; }
.kw-chip {
    display:inline-block;
    background:rgba(144,202,249,0.22); border:1px solid rgba(25,118,210,0.22);
    color:#1565C0; font-size:0.72rem; font-weight:600;
    padding:0.22rem 0.75rem; border-radius:999px; margin:0.2rem;
    transition:all 0.22s ease;
}
.kw-chip:hover { background:rgba(100,181,246,0.38); transform:scale(1.06) translateY(-1px); }

/* ── info box ── */
.info-box {
    background:rgba(227,242,253,0.8); border:1px solid rgba(33,150,243,0.2);
    border-radius:12px; padding:0.9rem 1.1rem;
    font-size:0.82rem; color:rgba(13,71,161,0.7);
    margin-bottom:1rem; display:flex; gap:0.6rem; align-items:flex-start;
}

/* ── llm sections ── */
.llm-section {
    background:rgba(255,255,255,0.68); border:1.5px solid rgba(33,150,243,0.16);
    border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1rem;
    box-shadow:0 2px 10px rgba(33,150,243,0.05); transition:all 0.26s ease;
}
.llm-section:hover { border-color:rgba(25,118,210,0.32); transform:translateY(-2px); box-shadow:0 6px 22px rgba(25,118,210,0.1); }
.llm-section-title {
    font-size:0.88rem; font-weight:700; color:#1565C0;
    margin-bottom:0.8rem; display:flex; align-items:center; gap:0.5rem;
    padding-bottom:0.6rem; border-bottom:1px solid rgba(33,150,243,0.12);
}
.llm-section-body { color:rgba(13,59,110,0.72); font-size:0.85rem; line-height:1.78; }
.llm-section-body strong { color:#0D47A1; }
.llm-section-body p  { margin:0.3rem 0; }
.llm-section-body ul, .llm-section-body ol { padding-left:1.3rem; margin:0.5rem 0; }
.llm-section-body li { margin-bottom:0.4rem; }

/* ── fancy divider ── */
.fdiv { height:1.5px; background:linear-gradient(90deg,transparent,rgba(33,150,243,0.3),transparent); margin:1rem 0; }

/* ── empty state ── */
.empty-state { text-align:center; padding:3rem 1rem; }
.empty-title { font-size:1.1rem; font-weight:700; color:#1565C0; margin-bottom:0.5rem; }
.empty-desc { color:rgba(21,101,192,0.55); font-size:0.85rem; }

/* ── feature cards ── */
.feat-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:0.8rem; margin-top:0.4rem; }
.feat-card {
    background:rgba(255,255,255,0.68); border:1.5px solid rgba(33,150,243,0.16);
    border-radius:16px; padding:1.1rem 1.2rem;
    box-shadow:0 2px 14px rgba(33,150,243,0.06); transition:all 0.28s ease;
}
.feat-card:hover { transform:translateY(-4px) scale(1.02); box-shadow:0 12px 32px rgba(25,118,210,0.14); border-color:rgba(25,118,210,0.32); }
.feat-num { font-size:0.58rem; font-weight:800; color:rgba(25,118,210,0.4); letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.35rem; display:block; }
.feat-title { font-size:0.87rem; font-weight:700; color:#0D47A1; margin-bottom:0.22rem; }
.feat-desc { font-size:0.76rem; color:rgba(13,59,110,0.56); line-height:1.5; }

/* ── footer ── */
.footer { text-align:center; color:rgba(25,118,210,0.3); font-size:0.73rem; margin-top:3.5rem; }

/* ── misc ── */
.stAlert { border-radius:14px !important; }
.stSpinner>div { color:#1976D2 !important; }

/* ── override Streamlit red accent globally ── */
:root { --primary-color:#1976D2; }

/* input/textarea wrapper — remove red border */
[data-baseweb="base-input"],
[data-baseweb="textarea"] {
    border-color:rgba(33,150,243,0.28) !important;
    box-shadow:none !important;
}
[data-baseweb="base-input"]:focus-within,
[data-baseweb="textarea"]:focus-within {
    border-color:#1976D2 !important;
    box-shadow:0 0 0 3px rgba(25,118,210,0.12) !important;
}
/* form border */
[data-testid="stForm"] {
    border-color:rgba(33,150,243,0.18) !important;
    border-radius:14px !important;
}
/* any button focus ring */
button:focus-visible { outline:2px solid #1976D2 !important; outline-offset:2px !important; }
/* native select/input focus */
select:focus,input:focus { border-color:#1976D2 !important; box-shadow:0 0 0 3px rgba(25,118,210,0.12) !important; }

/* ── tooltip ── */
[data-tip] { position:relative; cursor:help; }
[data-tip]::after {
    content:attr(data-tip);
    position:absolute; bottom:calc(100% + 8px); left:50%;
    transform:translateX(-50%);
    background:rgba(13,59,110,0.92); color:white;
    font-size:0.67rem; font-weight:500; letter-spacing:0.02em;
    padding:0.3rem 0.8rem; border-radius:8px;
    white-space:nowrap; opacity:0; pointer-events:none;
    transition:opacity 0.18s ease; z-index:9999;
    box-shadow:0 4px 14px rgba(0,0,0,0.22);
}
[data-tip]:hover::after { opacity:1; }

/* ── popover button ── */
[data-testid="stPopover"] button {
    background:rgba(227,242,253,0.85) !important;
    border:1px solid rgba(33,150,243,0.3) !important;
    border-radius:10px !important; color:#1565C0 !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:0.75rem !important; font-weight:600 !important;
    padding:0.22rem 0.75rem !important;
    transition:all 0.2s ease !important; box-shadow:none !important;
}
[data-testid="stPopover"] button:hover {
    background:rgba(144,202,249,0.45) !important;
    border-color:rgba(25,118,210,0.5) !important;
    transform:translateY(-1px) !important;
}

/* ── popover panel (content area) ── */
[data-baseweb="popover"] > div,
[data-baseweb="popover"] [role="dialog"] {
    background:linear-gradient(145deg,rgba(227,242,253,0.97) 0%,rgba(240,248,255,0.99) 100%) !important;
    border:1.5px solid rgba(33,150,243,0.28) !important;
    border-radius:20px !important;
    box-shadow:0 16px 56px rgba(25,118,210,0.22),0 2px 10px rgba(0,0,0,0.05) !important;
    backdrop-filter:blur(18px) !important;
    overflow-x:hidden !important;
    overflow-y:auto !important;
    max-height:80vh !important;
    min-width:540px !important;
    max-width:700px !important;
}
[data-testid="stPopoverBody"] {
    font-family:'Plus Jakarta Sans',sans-serif !important;
    padding:2rem 2.4rem !important;
    min-width:520px !important;
}

/* ── popover content components ── */
.pop-header {
    display:flex; align-items:center; justify-content:space-between;
    gap:1rem; margin-bottom:1.5rem;
    padding-bottom:1.1rem;
    border-bottom:1.5px solid rgba(33,150,243,0.18);
}
.pop-title { font-size:1rem; font-weight:800; color:#0D47A1; line-height:1.4; }
.pop-score {
    background:linear-gradient(135deg,rgba(144,202,249,0.35),rgba(187,222,251,0.5));
    border:1px solid rgba(33,150,243,0.28);
    color:#1565C0; font-size:0.72rem; font-weight:700;
    padding:0.28rem 0.9rem; border-radius:999px; flex-shrink:0;
    white-space:nowrap;
}
.pop-body { font-family:'Plus Jakarta Sans',sans-serif; }
.cpmk-title {
    font-size:0.82rem; font-weight:700; color:#1565C0;
    margin:0 0 1.1rem; line-height:1.65;
    padding:0.75rem 1rem;
    background:rgba(219,234,254,0.45); border-radius:10px;
}
.cpmk-item {
    display:flex; gap:1rem; align-items:flex-start;
    background:rgba(144,202,249,0.14); border:1px solid rgba(33,150,243,0.18);
    border-radius:12px; padding:1rem 1.1rem; margin-bottom:0.7rem;
}
.cpmk-lbl {
    font-size:0.65rem; font-weight:800; color:#1976D2;
    background:rgba(187,222,251,0.7); border:1px solid rgba(33,150,243,0.28);
    border-radius:6px; padding:0.2rem 0.6rem;
    white-space:nowrap; flex-shrink:0; line-height:2;
    letter-spacing:0.04em;
}
.cpmk-txt {
    font-size:0.83rem; color:rgba(13,59,110,0.82);
    line-height:1.75; padding-top:0.12rem;
}
.cpmk-body {
    font-size:0.81rem; color:rgba(13,59,110,0.65);
    line-height:1.78; margin:0.9rem 0 0;
    padding:1rem 1.2rem;
    background:rgba(240,248,255,0.8); border-radius:10px;
    border-left:3px solid rgba(33,150,243,0.3);
}
.pop-section-lbl {
    font-size:0.65rem; font-weight:800; letter-spacing:0.12em;
    text-transform:uppercase; color:rgba(25,118,210,0.5);
    margin-bottom:0.65rem;
}
.pub-item {
    font-size:0.83rem; color:rgba(13,59,110,0.8); line-height:1.68;
    padding:0.85rem 1.1rem; margin-bottom:0.5rem;
    background:rgba(144,202,249,0.13); border:1px solid rgba(33,150,243,0.15);
    border-radius:10px; border-left:3px solid rgba(25,118,210,0.38);
}
.pub-empty {
    font-size:0.78rem; color:rgba(21,101,192,0.38);
    font-style:italic; padding:0.5rem 0;
}
[data-testid="stPopoverBody"] p,
[data-testid="stPopoverBody"] li,
[data-testid="stPopoverBody"] span,
[data-testid="stPopoverBody"] .stMarkdown p {
    color:rgba(13,59,110,0.82) !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:0.84rem !important;
}
/* ensure inner element containers don't add extra padding */
[data-testid="stPopoverBody"] .element-container,
[data-testid="stPopoverBody"] .stMarkdown {
    margin-bottom:0 !important;
    padding:0 !important;
}
[data-testid="stPopoverBody"] strong {
    color:#0D47A1 !important; font-weight:700 !important;
}
[data-testid="stPopoverBody"] em { color:rgba(21,101,192,0.7) !important; }
[data-testid="stPopoverBody"] hr {
    border:none !important;
    border-top:1px solid rgba(33,150,243,0.18) !important;
    margin:0.5rem 0 !important;
}
[data-testid="stPopoverBody"] code {
    background:rgba(144,202,249,0.28) !important;
    color:#1565C0 !important;
    border:1px solid rgba(33,150,243,0.22) !important;
    border-radius:7px !important;
    padding:0.12rem 0.5rem !important;
    font-size:0.8rem !important;
}
[data-testid="stPopoverBody"] pre,
[data-testid="stPopoverBody"] .stCodeBlock {
    background:rgba(219,234,254,0.5) !important;
    border:1px solid rgba(33,150,243,0.2) !important;
    border-radius:12px !important;
}
[data-testid="stPopoverBody"] pre code {
    background:transparent !important;
    border:none !important;
    color:#1565C0 !important;
}
[data-testid="stPopoverBody"] h1,[data-testid="stPopoverBody"] h2,
[data-testid="stPopoverBody"] h3,[data-testid="stPopoverBody"] h4 {
    color:#0D47A1 !important; font-family:'Plus Jakarta Sans',sans-serif !important;
}
[data-testid="stPopoverBody"] .stTextPre {
    background:rgba(219,234,254,0.4) !important;
    color:rgba(13,59,110,0.8) !important;
    border:1px solid rgba(33,150,243,0.18) !important;
    border-radius:10px !important; padding:0.7rem !important;
    font-size:0.76rem !important; line-height:1.6 !important;
}

/* ── form submit button ── */
[data-testid="stFormSubmitButton"] > button {
    background:linear-gradient(135deg,#1565C0,#1976D2,#42A5F5) !important;
    background-size:200% auto !important;
    border:none !important; border-radius:14px !important;
    color:white !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:1rem !important; font-weight:700 !important;
    padding:0.8rem 2rem !important; letter-spacing:0.03em !important;
    transition:all 0.3s ease !important;
    width:100% !important;
    animation:btnPulse 2.8s ease-in-out infinite !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    transform:translateY(-3px) scale(1.02) !important;
    box-shadow:0 10px 32px rgba(25,118,210,0.5) !important;
    animation:none !important;
}
[data-testid="stFormSubmitButton"] > button:active {
    transform:translateY(0) scale(0.98) !important;
}


/* ── welcome page ── */
.welcome-screen {
    min-height:80vh; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    text-align:center; position:relative;
    overflow:hidden; padding:3rem 1rem 2rem;
}
/* floating shapes */
.wshape {
    position:fixed; border-radius:50%;
    pointer-events:none; z-index:0;
}
.ws1 {
    width:480px; height:480px; top:-120px; right:-100px;
    background:radial-gradient(circle,rgba(100,181,246,0.18) 0%,transparent 68%);
    animation:floatAround 9s ease-in-out infinite;
}
.ws2 {
    width:360px; height:360px; bottom:-90px; left:-80px;
    background:radial-gradient(circle,rgba(144,202,249,0.22) 0%,transparent 68%);
    animation:floatAround 12s ease-in-out infinite reverse;
}
.ws3 {
    width:240px; height:240px; top:38%; left:6%;
    background:radial-gradient(circle,rgba(33,150,243,0.1) 0%,transparent 68%);
    animation:floatAround 7s ease-in-out infinite; animation-delay:-3s;
}
.ws4 {
    width:190px; height:190px; top:18%; right:10%;
    background:radial-gradient(circle,rgba(66,165,245,0.13) 0%,transparent 68%);
    animation:floatAround 10s ease-in-out infinite reverse; animation-delay:-5s;
}
/* welcome content */
.wc-wrap { position:relative; z-index:1; max-width:580px; }
.wc-badge {
    display:inline-flex; align-items:center;
    background:rgba(187,222,251,0.45); border:1px solid rgba(33,150,243,0.3);
    color:#1565C0; font-size:0.68rem; font-weight:700;
    letter-spacing:0.14em; text-transform:uppercase;
    padding:0.28rem 1rem; border-radius:999px; margin-bottom:1.8rem;
    animation:fadeInUp 0.4s ease both;
}
.wc-title {
    font-size:clamp(3.5rem,8vw,5.5rem); font-weight:800; line-height:1.02;
    background:linear-gradient(135deg,#0D47A1 0%,#1976D2 40%,#42A5F5 70%,#90CAF9 100%);
    background-size:200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:fadeInUp 0.52s ease both, shimmer 5s linear infinite;
    margin:0 0 0.7rem;
}
.wc-tagline {
    font-size:1.02rem; color:rgba(13,71,161,0.55);
    line-height:1.8; margin-bottom:2.8rem;
    animation:fadeInUp 0.65s ease both;
}
.wc-divider {
    width:60px; height:1px; margin:0 auto 2.2rem;
    background:linear-gradient(90deg,transparent,rgba(25,118,210,0.4),transparent);
    animation:fadeInUp 0.72s ease both;
}
.wc-footer {
    text-align:center; color:rgba(21,101,192,0.35);
    font-size:0.74rem; margin-top:2.5rem; line-height:1.8;
    animation:fadeInUp 0.9s ease both;
}
.wc-footer strong { color:rgba(21,101,192,0.52); }
</style>
"""

# ── helpers ─────────────────────────────────────────────────────────────────

def initials(name: str) -> str:
    parts = name.strip().split()
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()

def donut_svg(value: float, max_val: float = 1.0, r: int = 36) -> str:
    pct = min(value / max_val, 1.0)
    circ = 2 * 3.14159 * r
    dash = pct * circ
    return f"""
    <div class="donut-ring">
      <svg width="90" height="90" viewBox="0 0 90 90">
        <circle cx="45" cy="45" r="{r}" fill="none" stroke="rgba(144,202,249,0.25)" stroke-width="8"/>
        <circle cx="45" cy="45" r="{r}" fill="none"
          stroke="url(#pg)" stroke-width="8" stroke-linecap="round"
          stroke-dasharray="{dash:.1f} {circ:.1f}"/>
        <defs>
          <linearGradient id="pg" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#90CAF9"/>
            <stop offset="100%" stop-color="#1565C0"/>
          </linearGradient>
        </defs>
      </svg>
      <div class="donut-center">
        <span class="donut-num">{value:.2f}</span>
        <span class="donut-sub">skor</span>
      </div>
    </div>"""

def score_bar(label: str, value: float) -> str:
    pct = min(value * 100, 100)
    return f"""
    <div style="margin:0.4rem 0;">
      <div style="display:flex;justify-content:space-between;font-size:0.73rem;color:rgba(21,101,192,0.6);margin-bottom:0.25rem;">
        <span>{label}</span><span>{value:.3f}</span>
      </div>
      <div style="background:rgba(144,202,249,0.25);border-radius:999px;height:5px;overflow:hidden;">
        <div style="width:{pct:.1f}%;height:100%;border-radius:999px;background:linear-gradient(90deg,#90CAF9,#1565C0);"></div>
      </div>
    </div>"""

def tags_html(items: list) -> str:
    return '<div class="tag-wrap">' + "".join(f'<span class="tag">{t}</span>' for t in items) + "</div>"

STATUS_META = {
    "Sangat Sesuai": ("✓",  "sb-green",  "Topik sangat sesuai dengan kurikulum Prodi Sains Data."),
    "Sesuai":        ("→",  "sb-blue",   "Topik sesuai dengan kurikulum Prodi Sains Data."),
    "Perlu Revisi":  ("!",  "sb-orange", "Topik perlu diperbaiki agar lebih sesuai kurikulum."),
    "Tidak Sesuai":  ("✗",  "sb-red",    "Topik tidak terdeteksi dalam domain Sains Data."),
}

def status_banner(status: str) -> str:
    icon, cls, desc = STATUS_META.get(status, ("?", "sb-blue", ""))
    return f"""
    <div class="status-banner {cls}">
      <div class="sb-icon">{icon}</div>
      <div>
        <div class="sb-title">{status}</div>
        <div class="sb-desc">{desc}</div>
      </div>
    </div>"""

def _inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
    return text

def _md2html(text: str) -> str:
    lines = text.split("\n")
    html: list[str] = []
    ctx = None
    blank_seen = False

    for line in lines:
        s = line.strip()
        if not s or re.match(r"^-{3,}$", s):
            if ctx in ("ul", "ol"):
                blank_seen = True
            continue
        m = re.match(r"^(\d+)\.\s+(.*)", s)
        if m:
            if ctx == "ul":
                html.append("</ul>"); ctx = None
            blank_seen = False
            if ctx != "ol":
                html.append("<ol>"); ctx = "ol"
            html.append(f"<li>{_inline(m.group(2))}</li>")
            continue
        if s.startswith("- ") or s.startswith("* "):
            content = _inline(s[2:])
            if ctx == "ol":
                if blank_seen:
                    html.append("</ol>"); blank_seen = False; ctx = "ul"
                    html.append("<ul>"); html.append(f"<li>{content}</li>")
                else:
                    html.append(f'<p style="margin:0.1rem 0 0.1rem 1rem;color:rgba(13,59,110,0.62);font-size:0.82rem;">{content}</p>')
            else:
                blank_seen = False
                if ctx != "ul":
                    html.append("<ul>"); ctx = "ul"
                html.append(f"<li>{content}</li>")
            continue
        if ctx == "ul": html.append("</ul>")
        elif ctx == "ol": html.append("</ol>")
        ctx = None; blank_seen = False
        html.append(f"<p>{_inline(s)}</p>")

    if ctx == "ul": html.append("</ul>")
    elif ctx == "ol": html.append("</ol>")
    return "\n".join(html)

def _clean_pdf_text(text: str) -> str:
    """Join PDF line-wrap artifacts; keep CPMK and numbered lines as their own blocks."""
    lines = text.split("\n")
    out: list[str] = []
    for raw in lines:
        s = raw.strip()
        is_header = bool(re.match(r"^(CPMK\d*\s*:|^\d+\.\s)", s))
        if not s:
            out.append("")
        elif is_header:
            out.append("\n" + s)
        elif out and out[-1] and out[-1] != "":
            out[-1] += " " + s
        else:
            out.append(s)
    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def _cpmk_html(text: str) -> str:
    """Render cleaned CPMK text as styled HTML for the popover."""
    parts = re.split(r"(\n\n+)", _clean_pdf_text(text))
    html = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if re.match(r"^CPMK\d*\s*:", s):
            label, _, rest = s.partition(":")
            html.append(
                f'<div class="cpmk-item">'
                f'<span class="cpmk-lbl">{label.strip()}</span>'
                f'<span class="cpmk-txt">{rest.strip()}</span>'
                f'</div>'
            )
        elif re.match(r"^\d+\.\s", s):
            html.append(f'<p class="cpmk-title">{s}</p>')
        else:
            html.append(f'<p class="cpmk-body">{s}</p>')
    return "\n".join(html)


def parse_llm_sections(text: str) -> list[dict]:
    parts = re.split(r"(##\s*\d+\..*)", text)
    sections = []
    for i in range(1, len(parts), 2):
        header = re.sub(r"##\s*\d+\.\s*", "", parts[i]).strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        m = re.search(r"(\d+)", parts[i])
        num = m.group(1) if m else "?"
        sections.append({"num": num, "title": header, "body": body})
    if not sections:
        sections.append({"num": "—", "title": "Analisis AI", "body": text})
    return sections

# ── backend ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Menyiapkan sistem, harap tunggu...")
def load_resources():
    groq_client = Groq(api_key=GROQ_API_KEY)
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    with open(DOSEN_JSON, "r", encoding="utf-8") as f:
        data_dosen = json.load(f)
    dosen_docs = _build_dosen_docs(data_dosen)
    dosen_store = QdrantVectorStore.from_documents(
        documents=dosen_docs, embedding=emb,
        location=":memory:", collection_name="dosen"
    )
    reader = PdfReader(KURIKULUM_PDF)
    full_text = "".join(p.extract_text() or "" for p in reader.pages)
    start = full_text.find("DESKRIPSI MATA KULIAH")
    sections = [s.strip() for s in re.split(r"(?=\n\d+\.\s+[A-ZA-Za-z])", full_text[start:]) if "CPMK" in s]
    kurikulum_docs = [Document(page_content=s, metadata={"mata_kuliah": _matkul(s)}) for s in sections]
    kurikulum_store = QdrantVectorStore.from_documents(
        documents=kurikulum_docs, embedding=emb,
        location=":memory:", collection_name="kurikulum"
    )
    return groq_client, emb, data_dosen, dosen_docs, dosen_store, kurikulum_store


def _build_dosen_docs(data_dosen):
    manual = {"MUHAMMAD FAHMY NADHIF": ["Explainable Artificial Intelligence","Deep Learning","Forecasting","Machine Learning"]}
    docs = []
    for d in data_dosen:
        nama = d.get("nama_dosen", "")
        bidang = d.get("bidang_keahlian", []) or manual.get(nama, [])
        pub = [a["judul"] for _, lst in d.get("daftar_publikasi", {}).items() for a in lst if a.get("judul")]
        pen = [p["judul"] for p in d.get("daftar_penelitian", []) if p.get("judul")]
        docs.append(Document(
            page_content=f"Nama: {nama}\nBidang: {', '.join(bidang)}\nPublikasi: {' | '.join(pub)}\nPenelitian: {' | '.join(pen)}",
            metadata={"nama_dosen": nama, "bidang_keahlian": bidang,
                      "publikasi_text": " ".join(pub), "penelitian_text": " ".join(pen)}
        ))
    return docs


def _matkul(text):
    m = re.search(r"\d+\.\s+(.*)", text)
    return m.group(1).strip() if m else "Unknown"


def _groq(client, prompt):
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return r.choices[0].message.content


def _sim(emb, a, b):
    return cosine_similarity([emb.embed_query(a)], [emb.embed_query(b)])[0][0]


def _expand(client, query):
    return _groq(client, f"Pakar Sains Data. Topik: {query}\nEkstrak 5-10 kata kunci akademik, pisah koma, tanpa penjelasan.")


def _rank_dosen(emb, docs, store, query, k=5):
    results = store.similarity_search_with_score(query, k=len(docs))
    ranked = []
    for doc, ret in results:
        exp = max((_sim(emb, query, s) for s in doc.metadata.get("bidang_keahlian", [])), default=0)
        pub = _sim(emb, query, doc.metadata.get("publikasi_text","") + " " + doc.metadata.get("penelitian_text","")) if doc.metadata.get("publikasi_text") else exp
        ranked.append({"nama": doc.metadata["nama_dosen"], "retrieval": ret,
                       "expertise": exp, "publication": pub,
                       "final": 0.35*ret + 0.40*exp + 0.25*pub, "document": doc})
    return sorted(ranked, key=lambda x: x["final"], reverse=True)[:k]


def _status(score):
    if score >= 0.40: return "Sangat Sesuai"
    if score >= 0.30: return "Sesuai"
    if score >= 0.20: return "Perlu Revisi"
    return "Tidak Sesuai"


def analisis(client, emb, dosen_docs, dosen_store, kurikulum_store, query):
    is_sd = "YA" in _groq(client, f"Topik ini termasuk Sains Data/ML/AI/NLP/Big Data?\nTopik: {query}\nJawab hanya: YA atau TIDAK").upper()
    if not is_sd:
        return {"query": query, "status": "Tidak Sesuai", "skor": 0, "kurikulum": []}
    expanded = _expand(client, query)
    kurikulum = kurikulum_store.similarity_search_with_score(query + " " + expanded, k=3)
    skor = max(s for _, s in kurikulum)
    dosen = _rank_dosen(emb, dosen_docs, dosen_store, query)
    return {
        "query": query, "expanded": expanded,
        "status": _status(skor) if skor >= 0.20 else "Perlu Revisi",
        "skor": skor, "kurikulum": kurikulum,
        "utama": dosen[0], "alternatif": dosen[1:3],
    }


def build_prompt(hasil):
    u = hasil.get("utama")
    if not u:
        return f"# ROLE\nEvaluator Prodi Sains Data.\n# TOPIK\n{hasil['query']}\n# STATUS\n{hasil['status']}\nJelaskan ketidaksesuaian. Jangan gunakan pengetahuan umum."
    prompt = f"# ROLE\nSistem rekomendasi dosen pembimbing skripsi Prodi Sains Data.\n# TOPIK\n{hasil['query']}\n# STATUS\n{hasil['status']}\n"
    if "expanded" in hasil: prompt += f"# KATA KUNCI\n{hasil['expanded']}\n\n"
    prompt += f"DOSEN UTAMA\nNama: {u['nama']}\nBidang: {u['document'].metadata['bidang_keahlian']}\nPublikasi: {u['document'].metadata['publikasi_text'][:700]}\nPenelitian: {u['document'].metadata['penelitian_text'][:700]}\n\nDOSEN ALTERNATIF\n"
    for d in hasil["alternatif"]:
        prompt += f"Nama: {d['nama']}\nBidang: {d['document'].metadata['bidang_keahlian']}\nPublikasi: {d['document'].metadata['publikasi_text'][:500]}\nPenelitian: {d['document'].metadata['penelitian_text'][:500]}\n\n"
    prompt += "KURIKULUM\n"
    for doc, sc in hasil["kurikulum"]:
        prompt += f"Mata Kuliah: {doc.metadata.get('mata_kuliah','?')}\nSkor: {sc:.3f}\n{doc.page_content}\n\n"
    prompt += """
# OUTPUT FORMAT
## 1. STATUS KELAYAKAN TOPIK
## 2. KESESUAIAN DENGAN KURIKULUM
Maks 3 mata kuliah: nama, CPMK, kutipan, hubungan, alasan.
---
## 3. DOSEN PEMBIMBING UTAMA
Nama, bidang, publikasi relevan, penelitian relevan, alasan utama.
---
## 4. DOSEN ALTERNATIF
Dua dosen: bidang pendukung, publikasi relevan, penelitian relevan, keterbatasan.
---
## 5. SARAN PENYEMPURNAAN JUDUL
2-3 alternatif judul + alasan singkat.

# CONSTRAINT
- Gunakan HANYA data yang diberikan. Nama dosen sama persis.
- Jangan tambah gelar, jabatan, bidang, atau publikasi baru.
- CPMK hanya jika ditemukan. Jika tidak ada: "Tidak ditemukan informasi pendukung."
"""
    return prompt

# ── UI ──────────────────────────────────────────────────────────────────────

EXAMPLES = [
    "Klasifikasi sentimen ulasan produk menggunakan BERT",
    "Prediksi harga rumah dengan XGBoost dan feature engineering",
    "Deteksi anomali pada data IoT menggunakan Autoencoder",
    "Sistem rekomendasi film berbasis collaborative filtering",
]


def welcome_page(n_dosen: int):
    # Floating animated shapes
    st.markdown("""
    <div class="ws1 wshape"></div>
    <div class="ws2 wshape"></div>
    <div class="ws3 wshape"></div>
    <div class="ws4 wshape"></div>
    <div class="welcome-screen">
      <div class="wc-wrap">
        <div class="wc-badge">Prodi Sains Data &nbsp;&middot;&nbsp; NLP Kelompok 6</div>
        <div class="wc-title">ThesisScope</div>
        <p class="wc-tagline">
          Validasi topik skripsimu dan temukan dosen pembimbing yang tepat
          menggunakan kecerdasan buatan berbasis data nyata.
        </p>
        <div class="wc-divider"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1.5, 2, 1.5])
    with col_btn:
        if st.button("Mulai Analisis", type="primary", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    st.markdown(f"""
    <div class="wc-footer">
      Dikembangkan oleh <strong>Kelompok 6</strong><br>
      Mata Kuliah Pemrosesan Bahasa Alami &nbsp;&middot;&nbsp; Prodi Sains Data<br>
      <span style="opacity:0.6;">{n_dosen} Dosen &nbsp;&middot;&nbsp; 34 Mata Kuliah &nbsp;&middot;&nbsp; Llama 3.3 70B</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="ThesisScope · Prodi Sains Data",
        page_icon="T", layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    groq_client, emb, data_dosen, dosen_docs, dosen_store, kurikulum_store = load_resources()
    n_dosen = len(data_dosen)

    # ── Welcome gate ──
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    if st.session_state.page == "welcome":
        welcome_page(n_dosen)
        return

    # ── Ambient blobs (main page) ──
    st.markdown('<div class="mb1"></div><div class="mb2"></div><div class="mb3"></div>', unsafe_allow_html=True)

    # ── Back button ──
    if st.button("← Kembali", key="back_btn"):
        st.session_state.page = "welcome"
        st.rerun()

    # ── Hero ──
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Find your</div>
      <div class="hero-title">ThesisScope</div>
      <p class="hero-sub">AI Research Advisor for Smart Thesis Planning</p>
      <div class="hero-stats">
        <div class="hero-stat"><span class="hero-stat-num">{n_dosen}</span><span class="hero-stat-label">Dosen Aktif</span></div>
        <div class="hero-stat"><span class="hero-stat-num">34</span><span class="hero-stat-label">Mata Kuliah</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input ──
    st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
    st.markdown('<p class="input-label">Masukkan Topik / Judul Skripsimu</p>', unsafe_allow_html=True)
    with st.form("form"):
        topik = st.text_area(
            label="topik", label_visibility="collapsed",
            placeholder="Contoh: Implementasi LSTM untuk prediksi harga saham menggunakan data historis IHSG...",
            height=90,
        )
        submitted = st.form_submit_button("Analisis Topik Sekarang", type="primary", use_container_width=True)
    col_hint, col_pop = st.columns([5, 1])
    with col_hint:
        st.markdown('<p class="input-hint">Semakin spesifik topikmu, semakin akurat rekomendasinya.</p>', unsafe_allow_html=True)
    with col_pop:
        with st.popover("Panduan"):
            st.markdown(
                '<div class="pop-header">'
                '<span class="pop-title">Formula Topik Ideal</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pop-body">'
                '<div class="pop-section-lbl">Struktur</div>'
                '<div class="pub-item" style="font-family:monospace;font-weight:700;">'
                '[Metode] + [Objek Data] + [Tujuan/Konteks]'
                '</div>'
                '<div class="pop-section-lbl" style="margin-top:1rem;">Contoh Topik Baik</div>'
                '<div class="pub-item">Implementasi <strong>LSTM</strong> untuk prediksi harga saham IHSG</div>'
                '<div class="pub-item">Klasifikasi sentimen ulasan produk menggunakan <strong>BERT</strong></div>'
                '<div class="pub-item">Deteksi anomali data IoT dengan <strong>Autoencoder</strong></div>'
                '<div class="pub-item">Sistem rekomendasi film berbasis <strong>Collaborative Filtering</strong></div>'
                '<div class="pop-section-lbl" style="margin-top:1rem;">Tips Penulisan</div>'
                '<div class="pub-item">Cantumkan nama metode atau algoritma secara eksplisit</div>'
                '<div class="pub-item">Sebutkan objek atau domain data secara spesifik</div>'
                '<div class="pub-item">Tambahkan tujuan atau konteks penelitian</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="example-wrap">' + "".join(f'<span class="example-pill">"{e}"</span>' for e in EXAMPLES) + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and not topik.strip():
        st.warning("Topik belum diisi. Silakan masukkan topik skripsimu terlebih dahulu.")
        return

    if not submitted:
        st.markdown("""
        <div class="fdiv"></div>
        <div class="feat-grid">
          <div class="feat-card">
            <span class="feat-num">01</span>
            <div class="feat-title">Validasi Kurikulum</div>
            <div class="feat-desc">Topikmu dicocokkan dengan 34 mata kuliah di kurikulum Prodi Sains Data secara semantik.</div>
          </div>
          <div class="feat-card">
            <span class="feat-num">02</span>
            <div class="feat-title">Rekomendasi Dosen</div>
            <div class="feat-desc">Dosen diranking berdasarkan 3 faktor: keahlian, publikasi, dan penelitian.</div>
          </div>
          <div class="feat-card">
            <span class="feat-num">03</span>
            <div class="feat-title">Analisis AI</div>
            <div class="feat-desc">Memberikan penjelasan mendalam dan saran alternatif judul yang lebih akademik.</div>
          </div>
          <div class="feat-card">
            <span class="feat-num">04</span>
            <div class="feat-title">Berbasis Data Nyata</div>
            <div class="feat-desc">Seluruh proses menggunakan data SINTA dosen dan kurikulum resmi.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="footer">NLP-Kelompok6</div>', unsafe_allow_html=True)
        return

    # ── Step Tracker (single placeholder, updated in-place) ──
    _step = st.empty()
    _step.markdown("""
    <div class="steps-wrap">
      <div class="step step-done">Input Topik</div>
      <div class="step-line step-line-done"></div>
      <div class="step step-active">Analisis</div>
      <div class="step-line"></div>
      <div class="step step-pending">Hasil</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Sedang memproses topikmu..."):
        hasil = analisis(groq_client, emb, dosen_docs, dosen_store, kurikulum_store, topik)

    _step.markdown("""
    <div class="steps-wrap">
      <div class="step step-done">Input Topik</div>
      <div class="step-line step-line-done"></div>
      <div class="step step-done">Analisis</div>
      <div class="step-line step-line-done"></div>
      <div class="step step-active">Hasil</div>
    </div>
    """, unsafe_allow_html=True)

    status = hasil["status"]
    st.markdown(status_banner(status), unsafe_allow_html=True)

    if "expanded" in hasil:
        kws = [k.strip() for k in hasil["expanded"].split(",") if k.strip()]
        chips = "".join(f'<span class="kw-chip">{k}</span>' for k in kws[:10])
        st.markdown(f'<div class="kw-section"><div class="kw-label">Kata Kunci Terdeteksi</div>{chips}</div>', unsafe_allow_html=True)

    if status == "Tidak Sesuai" and "utama" not in hasil:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-title">Topik di luar domain Sains Data</div>
          <div class="empty-desc">Coba perjelas topikmu dengan menambahkan kata kunci seperti<br>
          <em>machine learning, data, prediksi, klasifikasi, NLP,</em> atau <em>analitik.</em></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="footer">NLP-Kelompok6</div>', unsafe_allow_html=True)
        return

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["Kurikulum", "Dosen", "Analisis"])

    # ── TAB 1: Kurikulum ──
    with tab1:
        col_d, col_s = st.columns([2.5, 1], gap="large")
        with col_d:
            st.markdown('<div class="sec-head">Mata Kuliah yang Paling Relevan</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">Mata kuliah di bawah ini dipilih secara semantik berdasarkan kecocokan makna — bukan sekadar kesamaan kata.</div>', unsafe_allow_html=True)
            for doc, sc in hasil.get("kurikulum", []):
                mk = doc.metadata.get("mata_kuliah", "—")
                pct = min(sc / 0.5 * 100, 100)
                preview = doc.page_content[:300].replace("\n", "<br>")
                st.markdown(f"""
                <div class="mk-card">
                  <div class="mk-header">
                    <div class="mk-name">{mk}</div>
                    <span class="mk-score-pill">{sc:.3f}</span>
                  </div>
                  <div class="mk-bar-track"><div class="mk-bar-fill" style="width:{pct:.0f}%"></div></div>
                  <div class="mk-body">{preview}...</div>
                </div>
                """, unsafe_allow_html=True)
                with st.popover(f"Detail CPMK — {mk}"):
                    st.markdown(
                        f'<div class="pop-header">'
                        f'<span class="pop-title">{mk}</span>'
                        f'<span class="pop-score">{sc:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="pop-body">{_cpmk_html(doc.page_content)}</div>',
                        unsafe_allow_html=True,
                    )
        with col_s:
            st.markdown('<div class="sec-head">Skor</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;margin-bottom:1rem;">
              <div class="donut-wrap">
                {donut_svg(hasil.get('skor', 0))}
                <div class="donut-label">Kesesuaian Kurikulum</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(255,255,255,0.62);border:1.5px solid rgba(33,150,243,0.18);border-radius:14px;padding:1rem 1.2rem;font-size:0.78rem;color:rgba(13,71,161,0.7);">
              <div style="font-weight:700;margin-bottom:0.6rem;color:#1565C0;">Panduan Skor</div>
              <div style="margin-bottom:0.3rem;">&#9679; &ge; 0.40 &rarr; Sangat Sesuai</div>
              <div style="margin-bottom:0.3rem;">&#9679; &ge; 0.30 &rarr; Sesuai</div>
              <div style="margin-bottom:0.3rem;">&#9679; &ge; 0.20 &rarr; Perlu Revisi</div>
              <div>&#9679; &lt; 0.20 &rarr; Tidak Sesuai</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: Dosen ──
    with tab2:
        if "utama" not in hasil:
            st.markdown('<div class="empty-state"><div class="empty-title">Tidak ada dosen yang direkomendasikan</div></div>', unsafe_allow_html=True)
        else:
            col_u, col_a = st.columns([1, 1], gap="large")
            with col_u:
                st.markdown('<div class="sec-head">Pembimbing Utama</div>', unsafe_allow_html=True)
                u = hasil["utama"]
                b = u["document"].metadata.get("bidang_keahlian", [])
                st.markdown(f"""
                <div class="dosen-utama">
                  <span class="utama-badge">REKOMENDASI UTAMA</span>
                  <div class="dosen-top">
                    <div class="avatar">{initials(u['nama'])}</div>
                    <div>
                      <div class="dosen-name">{u['nama'].title()}</div>
                      <div class="dosen-role">Dosen Prodi Sains Data</div>
                    </div>
                  </div>
                  {tags_html(b)}
                  {score_bar("Relevansi Total", u['final'])}
                  <div class="score-row">
                    <div class="score-item" data-tip="Kemiripan bidang keahlian dengan topik">
                      <span class="score-num">{u['expertise']:.2f}</span>
                      <div class="score-bar-mini"><div class="score-bar-mini-fill" style="width:{u['expertise']*100:.0f}%"></div></div>
                      <span class="score-lbl">Keahlian</span>
                    </div>
                    <div class="score-item" data-tip="Relevansi publikasi & penelitian dosen">
                      <span class="score-num">{u['publication']:.2f}</span>
                      <div class="score-bar-mini"><div class="score-bar-mini-fill" style="width:{u['publication']*100:.0f}%"></div></div>
                      <span class="score-lbl">Publikasi</span>
                    </div>
                    <div class="score-item" data-tip="Skor kemiripan dokumen (vector search)">
                      <span class="score-num">{u['retrieval']:.2f}</span>
                      <div class="score-bar-mini"><div class="score-bar-mini-fill" style="width:{u['retrieval']*100:.0f}%"></div></div>
                      <span class="score-lbl">Retrieval</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                _pc = u["document"].page_content
                _pub_m = re.search(r"Publikasi: (.*?)(?:\nPenelitian:|$)", _pc, re.DOTALL)
                _pen_m = re.search(r"Penelitian: (.*?)$", _pc, re.DOTALL)
                _pubs = [p.strip() for p in _pub_m.group(1).split(" | ") if p.strip()] if _pub_m else []
                _pens = [p.strip() for p in _pen_m.group(1).split(" | ") if p.strip()] if _pen_m else []
                with st.popover("Publikasi & Penelitian"):
                    st.markdown(
                        f'<div class="pop-header">'
                        f'<span class="pop-title">{u["nama"].title()}</span>'
                        f'<span class="pop-score">Dosen Utama</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    pub_items = "".join(f'<div class="pub-item">{p}</div>' for p in _pubs[:6]) or '<div class="pub-empty">Data tidak tersedia</div>'
                    pen_items = "".join(f'<div class="pub-item">{p}</div>' for p in _pens[:6]) or '<div class="pub-empty">Data tidak tersedia</div>'
                    st.markdown(
                        f'<div class="pop-body">'
                        f'<div class="pop-section-lbl">Publikasi</div>{pub_items}'
                        f'<div class="pop-section-lbl" style="margin-top:1rem;">Penelitian</div>{pen_items}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with col_a:
                st.markdown('<div class="sec-head">Dosen Alternatif</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">Dosen alternatif dapat menjadi pilihan jika dosen utama tidak tersedia atau topik memiliki irisan dengan bidang mereka.</div>', unsafe_allow_html=True)
                for i, d in enumerate(hasil.get("alternatif", []), 1):
                    b2 = d["document"].metadata.get("bidang_keahlian", [])
                    st.markdown(f"""
                    <div class="dosen-alt">
                      <div class="dosen-top">
                        <div class="avatar-sm">{initials(d['nama'])}</div>
                        <div>
                          <div class="dosen-name" style="font-size:0.92rem;">Alternatif #{i} &middot; {d['nama'].title()}</div>
                          <div class="dosen-role">Dosen Prodi Sains Data</div>
                        </div>
                      </div>
                      {tags_html(b2[:4])}
                      {score_bar("Relevansi", d['final'])}
                    </div>
                    """, unsafe_allow_html=True)
                    _pc2 = d["document"].page_content
                    _pub_m2 = re.search(r"Publikasi: (.*?)(?:\nPenelitian:|$)", _pc2, re.DOTALL)
                    _pen_m2 = re.search(r"Penelitian: (.*?)$", _pc2, re.DOTALL)
                    _pubs2 = [p.strip() for p in _pub_m2.group(1).split(" | ") if p.strip()] if _pub_m2 else []
                    _pens2 = [p.strip() for p in _pen_m2.group(1).split(" | ") if p.strip()] if _pen_m2 else []
                    with st.popover(f"Publikasi Alternatif #{i}"):
                        st.markdown(
                            f'<div class="pop-header">'
                            f'<span class="pop-title">{d["nama"].title()}</span>'
                            f'<span class="pop-score">Alternatif #{i}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        pub2_items = "".join(f'<div class="pub-item">{p}</div>' for p in _pubs2[:6]) or '<div class="pub-empty">Data tidak tersedia</div>'
                        pen2_items = "".join(f'<div class="pub-item">{p}</div>' for p in _pens2[:6]) or '<div class="pub-empty">Data tidak tersedia</div>'
                        st.markdown(
                            f'<div class="pop-body">'
                            f'<div class="pop-section-lbl">Publikasi</div>{pub2_items}'
                            f'<div class="pop-section-lbl" style="margin-top:1rem;">Penelitian</div>{pen2_items}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── TAB 3: Analisis AI ──
    with tab3:
        st.markdown(
            '<div class="info-box">'
            'Analisis ini dihasilkan oleh '
            'berdasarkan data dosen dan kurikulum — bukan pengetahuan umum.'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Sedang menyusun analisis lengkap..."):
            prompt = build_prompt(hasil)
            llm_out = _groq(groq_client, prompt)
        sections = parse_llm_sections(llm_out)
        for sec in sections:
            st.markdown(
                f'<div class="llm-section">'
                f'<div class="llm-section-title">{sec["num"]}. {sec["title"]}</div>'
                f'<div class="llm-section-body">{_md2html(sec["body"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="footer">NLP-Kelompok6</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

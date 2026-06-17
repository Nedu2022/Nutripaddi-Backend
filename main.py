from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import torch, timm, json, httpx, io, os, base64, asyncio
import json as _json
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def load_local_env(path=".env"):
    if not os.path.exists(path):
        return

    with open(path) as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- config ---
USDA_API_KEY = os.getenv("USDA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_AUTH_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "false").lower() in {"1", "true", "yes", "on"}
MAMA_URL = "https://api-inference.huggingface.co/models/HelpMumHQ/MamaBot-Llama"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

bearer_scheme = HTTPBearer(auto_error=False)



USDA_SEARCH = {
    # West Africa
    "jollof_rice": "jollof rice", "jollof_ghana": "rice tomato stew",
    "egusi_soup": "melon seed soup", "ogbono": "bush mango soup",
    "okro_soup": "okra soup", "ewa_agoyin": "black eyed peas cooked",
    "moin_moin": "bean pudding steamed", "akara_and_eko": "black eyed pea fritter",
    "suya": "grilled beef skewer spiced", "nkwobi": "spiced cow foot",
    "pepper_soup": "spiced meat broth", "fried_plantains_(dodo)": "fried plantains",
    "fried_plantain": "fried plantains", "asaro": "yam porridge",
    "amala_and_ewedu_gbegiri": "yam flour porridge", "boli(bole)": "roasted plantain",
    "eba": "cassava cooked", "semo": "semolina cooked", "pounded_yam": "yam cooked",
    "fufu": "cassava cooked", "waakye": "rice beans",
    "meat_pie": "meat pie", "chin_chin": "fried dough snack", "puff_puff": "fried dough ball",
    "abacha_and_ugba(african_salad)": "tapioca salad",

    "edikang_ikong_soup": "vegetable soup nigerian", "vegetable_soup": "vegetable soup nigerian",
    "ndole": "bitterleaf stew", "eru": "spinach stew", "ekwang": "cocoyam leaves",
    "palm_nut_soup": "palm oil soup",

    "doro_wat": "chicken stew", "doro": "chicken cooked", "kitfo": "beef minced",
    "shiro_wat": "chickpea stew", "tibs": "beef cooked", "injera": "teff flatbread",
    "tire_siga": "beef raw", "beyaynetu": "lentil vegetable stew",
    "firfir": "flatbread cooked", "genfo": "barley porridge", "kikil": "beef stew",
    "shekla_tibs": "beef grilled",

    "fried_chicken": "chicken fried", "chicken": "chicken cooked",
    "coleslaw": "coleslaw", "salad": "salad vegetable", "rice": "rice cooked white",
    "beef": "beef cooked", "fish": "fish cooked", "egg": "egg boiled", "beans": "beans cooked",
}


def is_supabase_configured():
    return bool(SUPABASE_URL and SUPABASE_AUTH_KEY)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if not AUTH_REQUIRED:
        return None

    if not is_supabase_configured():
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured")

    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": SUPABASE_AUTH_KEY,
                },
            )
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Could not verify authentication")

    if response.status_code >= 400:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")

    return response.json()


ckpt = torch.load("best.pt", map_location="cpu")
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    sd, classes = ckpt["state_dict"], ckpt.get("classes")
else:
    sd, classes = ckpt, None
if classes is None:
    with open("class_names.json") as f:
        classes = json.load(f)
clf = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
clf.load_state_dict(sd); clf.eval()
print(f"Loaded {len(classes)} classes")
tfm = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# --- USDA nutrition ---
async def get_nutrition(dish):
    if not USDA_API_KEY:
        return None

    q = USDA_SEARCH.get(dish, dish.replace("_", " "))
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.get("https://api.nal.usda.gov/fdc/v1/foods/search",
                            params={"query": q, "api_key": USDA_API_KEY, "pageSize": 1})
        foods = r.json().get("foods", [])
    except Exception:
        return None
    if not foods:
        return None
    food = foods[0]
    want = {1008:"calories",1003:"protein_g",1089:"iron_mg",1177:"folate_mcg",1190:"folate_mcg"}
    out = {"match": food["description"], "calories":0,"protein_g":0,"iron_mg":0,"folate_mcg":0}
    for n in food.get("foodNutrients", []):
        if n.get("nutrientId") in want:
            out[want[n["nutrientId"]]] = n.get("value", 0)
    return out

# --- simple, clear advice (no MamaBot, instant) ---
def build_advice(total):
    msgs = []
    if total["iron_mg"] < 4:
        msgs.append("This meal is low in iron. Iron helps your baby's blood and brain grow. Add beans, ugu (pumpkin leaf), liver, or egg.")
    if total["protein_g"] < 12:
        msgs.append("This meal is low in protein. Protein builds your baby's body. Add egg, fish, beans, or meat.")
    if total["folate_mcg"] < 100:
        msgs.append("This meal is low in folic acid. Folic acid protects your baby from birth defects. Add green vegetables or oranges.")
    if not msgs:
        return "This meal is well balanced. Keep eating protein, vegetables, and fruit with your food."
    return " ".join(msgs)

# --- translation (Igbo / Yoruba / Hausa via Gemini, English fallback) ---
LANG_NAMES = {"en":"English","ig":"Igbo","yo":"Yoruba","ha":"Hausa"}
async def translate(text, lang):
    if lang == "en" or lang not in LANG_NAMES:
        return text
    if not GEMINI_KEY:
        return text

    prompt = (f"Translate this maternal nutrition advice into {LANG_NAMES[lang]}. "
              f"Use simple everyday words an ordinary mother would understand. "
              f"Reply with only the translation, nothing else:\n\n{text}")
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{GEMINI_URL}?key={GEMINI_KEY}",
                json={"contents":[{"parts":[{"text":prompt}]}]})
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return text

# --- Gemini plate recognition (pan-African) ---
async def recognize_with_gemini(img_bytes):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_KEY is not configured")

    b64 = base64.b64encode(img_bytes).decode()
    prompt = ('List the distinct African indigenous foods on this plate. Reply ONLY with a JSON array of '
              'lowercase names with underscores, e.g. ["jollof_rice","doro_wat","injera","ekwang"]. No other text.')
    async with httpx.AsyncClient(timeout=25) as c:
        r = await c.post(f"{GEMINI_URL}?key={GEMINI_KEY}",
            json={"contents":[{"parts":[{"text":prompt},
                  {"inline_data":{"mime_type":"image/jpeg","data":b64}}]}]})
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return _json.loads(txt)

async def maternal_advice(total):
    base = build_advice(total)          # reliable rule-based advice
    if not HF_TOKEN:
        return base

    try:
        prompt = (f"Rewrite this maternal nutrition advice in warm, simple words for a "
                  f"pregnant mother. Keep the same meaning:\n\n{base}")
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(MAMA_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt, "parameters": {"max_new_tokens": 200}})
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"].replace(prompt, "").strip()
            if text:
                return text
    except Exception:
        pass
    return base

# --- endpoints ---
@app.post("/scan-plate")
async def scan_plate(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    current_user: dict | None = Depends(get_current_user),
):
    img_bytes = await file.read()
    try:
        names = await recognize_with_gemini(img_bytes)
    except Exception as e:
        return {"foods": [], "advice": "", "error": str(e)}
    nutritions = await asyncio.gather(*[get_nutrition(n) for n in names])
    foods, total = [], {"calories":0,"protein_g":0,"iron_mg":0,"folate_mcg":0}
    for n, nut in zip(names, nutritions):
        foods.append({"key": n, "name": n.replace("_"," ").title(), "nutrition": nut})
        if nut:
            for k in total: total[k] += nut[k]
    advice_en = build_advice(total)
    advice = await translate(advice_en, lang)
    return {"foods": foods, "total": total, "advice": advice, "advice_en": advice_en}

@app.post("/scan")   # your own model, single dish
async def scan(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    current_user: dict | None = Depends(get_current_user),
):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        p = F.softmax(clf(tfm(img).unsqueeze(0)), 1)[0]
    conf, idx = p.max(0)
    key = classes[int(idx)]
    nut = await get_nutrition(key)
    total = nut if nut else {"calories":0,"protein_g":0,"iron_mg":0,"folate_mcg":0}
    advice_en = build_advice(total)
    advice = await translate(advice_en, lang)
    return {"foods": [{"key": key, "name": key.replace("_"," ").title(),
                       "confidence": round(conf.item()*100), "nutrition": nut}],
            "advice": advice, "advice_en": advice_en}

@app.get("/")
def health():
    return {
        "status": "NutriPadi API running",
        "auth_required": AUTH_REQUIRED,
        "supabase_configured": is_supabase_configured(),
    }

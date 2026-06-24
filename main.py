from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import torch, timm, json, httpx, io, os, base64, asyncio, re, random
import json as _json
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path


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
MIN_CLASSIFIER_CONFIDENCE = float(os.getenv("MIN_CLASSIFIER_CONFIDENCE", "0.65"))

bearer_scheme = HTTPBearer(auto_error=False)

BASE_DIR = Path(__file__).resolve().parent
FOOD_CATALOG_PATH = BASE_DIR / "data" / "food_catalog.json"


def load_food_catalog(path=FOOD_CATALOG_PATH):
    with open(path) as f:
        catalog = json.load(f)

    if not isinstance(catalog, dict) or not catalog:
        raise RuntimeError("Food catalog must be a non-empty object")

    return catalog


def food_token(value):
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ").replace("+", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def build_food_aliases(catalog):
    aliases = {}
    conflicts = []
    for key, entry in catalog.items():
        labels = {key, entry.get("display_name", "")}
        labels.update(entry.get("aliases", []))
        for label in labels:
            token = food_token(label)
            if not token:
                continue

            existing = aliases.get(token)
            if existing and existing != key:
                conflicts.append(f"{token}: {existing} / {key}")
            else:
                aliases[token] = key

    if conflicts:
        raise RuntimeError("Food catalog alias conflicts: " + "; ".join(conflicts))

    return aliases


def validate_food_catalog(catalog, class_labels):
    missing = [label for label in class_labels if label not in catalog]
    if missing:
        raise RuntimeError(
            "Food catalog is missing classifier labels: " + ", ".join(missing)
        )


FOOD_CATALOG = load_food_catalog()
FOOD_ALIASES = build_food_aliases(FOOD_CATALOG)

FOOD_ORIGINS = {"african", "western", "asian", "middle_eastern", "mixed", "generic", "unknown"}
AFRICAN_REGIONS = {"Nigeria", "Ghana", "Kenya", "Ethiopia", "Cameroon", "West Africa"}
OPEN_FOOD_ORIGIN_HINTS = {
    "western": {
        "pizza", "burger", "hamburger", "cheeseburger", "pasta", "spaghetti",
        "lasagna", "sandwich", "hot_dog", "fries", "french_fries", "steak",
        "macaroni", "mac_and_cheese", "fried_chicken",
    },
    "asian": {
        "sushi", "ramen", "noodles", "fried_rice", "dumpling", "spring_roll",
        "curry", "biryani", "pad_thai", "teriyaki", "kimchi", "pho",
    },
    "middle_eastern": {
        "shawarma", "hummus", "falafel", "kebab", "kabob", "tabbouleh",
        "pita", "mansaf", "kofta", "baklava",
    },
    "mixed": {
        "mixed_plate", "combo", "platter", "rice_and_chicken", "rice_and_beans",
        "chicken_and_rice", "salad_bowl",
    },
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
validate_food_catalog(FOOD_CATALOG, classes)
clf = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
clf.load_state_dict(sd); clf.eval()
print(f"Loaded {len(classes)} classes")
tfm = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def normalize_food_key(value):
    token = food_token(value)
    return FOOD_ALIASES.get(token, token)


def food_display_name(key):
    normalized_key = normalize_food_key(key)
    entry = FOOD_CATALOG.get(normalized_key)
    if entry:
        return entry.get("display_name", normalized_key.replace("_", " ").title())

    return normalized_key.replace("_", " ").title()


def catalog_origin(entry):
    region = entry.get("region")
    if region in AFRICAN_REGIONS:
        return "african", region
    if region == "Generic":
        return "generic", None
    return "unknown", region


def infer_open_food_origin(key):
    token = food_token(key)
    if not token:
        return "unknown"

    for origin, hints in OPEN_FOOD_ORIGIN_HINTS.items():
        if token in hints or any(hint in token for hint in hints):
            return origin

    return "unknown"


def clean_food_origin(value):
    origin = food_token(value)
    if origin in {"middle_east", "middle_eastern", "middle_eastern_food"}:
        return "middle_eastern"
    if origin in {"foreign", "international"}:
        return "unknown"
    if origin in FOOD_ORIGINS:
        return origin
    return "unknown"


def low_detection_confidence(value):
    if value is None:
        return False
    if isinstance(value, str):
        return food_token(value) in {"low", "uncertain", "unsure", "not_sure"}
    try:
        confidence = float(value)
        if confidence > 1:
            confidence = confidence / 100
        return confidence < 0.5
    except (TypeError, ValueError):
        return False


def _parse_portion_g(value):
    try:
        g = int(float(value))
        return g if g > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_food_detection(value):
    if isinstance(value, dict):
        raw_name = (
            value.get("name")
            or value.get("key")
            or value.get("food")
            or value.get("food_name")
            or value.get("label")
        )
        origin = clean_food_origin(
            value.get("food_origin")
            or value.get("origin")
            or value.get("type")
            or value.get("cuisine")
        )
        country = value.get("country") or value.get("local_country")
        confidence = value.get("confidence")
        portion_g = value.get("portion_g")
        meal_role = value.get("meal_role")
    else:
        raw_name = value
        origin = "unknown"
        country = None
        confidence = None
        portion_g = None
        meal_role = None

    key = normalize_food_key(raw_name)
    if country:
        for candidate in (f"{raw_name} {country}", f"{country} {raw_name}"):
            candidate_key = normalize_food_key(candidate)
            if candidate_key in FOOD_CATALOG:
                key = candidate_key
                break
    if not key:
        key = "unknown_food"
        raw_name = "unknown food"

    entry = FOOD_CATALOG.get(key)
    if entry:
        catalog_food_origin, catalog_country = catalog_origin(entry)
        if catalog_food_origin != "generic":
            origin = catalog_food_origin
            country = catalog_country or country
        elif origin == "unknown":
            origin = infer_open_food_origin(key)
    elif origin == "unknown":
        origin = infer_open_food_origin(key)

    if key.startswith("unknown") or key in {"unclear_food", "unidentified_food"}:
        origin = "unknown"

    return {
        "key": key,
        "raw_name": str(raw_name or key).strip(),
        "food_origin": origin,
        "country": country,
        "confidence": confidence,
        "portion_g": _parse_portion_g(portion_g),
        "meal_role": meal_role if meal_role in {"main", "protein", "side", "drink", "condiment"} else None,
    }


def nutrition_query_for_food(dish):
    if isinstance(dish, dict):
        key = dish.get("key") or normalize_food_key(dish.get("raw_name"))
    else:
        key = normalize_food_key(dish)
    entry = FOOD_CATALOG.get(key)
    if entry:
        return entry.get("usda_query"), key, True

    raw_name = dish.get("raw_name") if isinstance(dish, dict) else dish
    query = str(raw_name or key or "").strip().replace("_", " ")
    if not query:
        return None, key, False

    return query, key, False


def food_response(
    key,
    nutrition,
    confidence=None,
    source=None,
    model_guess=None,
    needs_review=False,
):
    detection = normalize_food_detection(key)
    normalized_key = detection["key"]
    entry = FOOD_CATALOG.get(normalized_key, {})
    origin = detection["food_origin"]
    country = detection["country"]
    item = {
        "key": normalized_key,
        "name": food_display_name(normalized_key) if entry else food_display_name(detection["raw_name"]),
        "food_origin": origin,
        "mapped": bool(entry),
        "nutrition": nutrition,
    }
    if entry.get("region"):
        item["region"] = entry["region"]
    if country:
        item["country"] = country
    if confidence is not None:
        item["confidence"] = round(confidence * 100)
    elif detection.get("confidence") is not None:
        item["confidence"] = detection["confidence"]
    if source:
        item["source"] = source
    if model_guess:
        item["model_guess"] = model_guess
    if detection.get("portion_g"):
        item["portion_g"] = detection["portion_g"]
    if detection.get("meal_role"):
        item["meal_role"] = detection["meal_role"]
    detection_low_confidence = low_detection_confidence(detection.get("confidence"))
    if needs_review or origin == "unknown" or detection_low_confidence:
        item["needs_review"] = True
        if origin == "unknown":
            item["review_reason"] = "unknown_food"
        elif detection_low_confidence:
            item["review_reason"] = "low_detection_confidence"
        else:
            item["review_reason"] = "low_model_confidence"
    return item


def normalize_food_list(values):
    foods = []
    seen = set()
    if not isinstance(values, list):
        return foods

    for value in values:
        detection = normalize_food_detection(value)
        key = detection["key"]
        if not key or key in seen:
            continue

        seen.add(key)
        foods.append(detection)

    return foods


def vision_catalog_prompt():
    allowed_keys = ", ".join(f'"{key}"' for key in sorted(FOOD_CATALOG))
    return (
        "You are a food recognition expert. Identify EVERY distinct component visible on this plate "
        "— the main dish, all sides, proteins, sauces, and drinks, even if they share the same bowl. "
        "If the food is African, use the most correct country/local food name or one of these known "
        f"African catalog keys when it matches: [{allowed_keys}]. If the food is Western, Asian, "
        "Middle Eastern, mixed, or foreign, name it exactly as it is. For unclear items use "
        "unknown_food and food_origin unknown. "
        "For each food also estimate the portion_g (integer grams visible on the plate) and assign "
        "a meal_role: 'main' for the primary starch or base dish, 'protein' for meat/fish/eggs/beans, "
        "'side' for vegetables and accompaniments, 'drink' for beverages, 'condiment' for sauces/oils. "
        "Reply ONLY with a JSON array of objects: "
        '[{"name":"jollof_rice","food_origin":"african","country":"Nigeria","confidence":"high","portion_g":280,"meal_role":"main"},'
        '{"name":"fried_chicken","food_origin":"western","country":null,"confidence":"high","portion_g":150,"meal_role":"protein"},'
        '{"name":"coleslaw","food_origin":"western","country":null,"confidence":"medium","portion_g":80,"meal_role":"side"}]. '
        "Allowed food_origin: african, western, asian, middle_eastern, mixed, generic, unknown. "
        "Allowed meal_role: main, protein, side, drink, condiment. "
        "portion_g must be an integer greater than 0. confidence must be high, medium, or low. "
        "No markdown and no extra text."
    )


# --- USDA nutrition ---
async def get_nutrition(dish, portion_g=None):
    if not USDA_API_KEY:
        return None

    q, _, _ = nutrition_query_for_food(dish)
    if not q:
        return None

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
    # USDA values are per 100g — scale to the actual estimated portion
    if portion_g and portion_g > 0:
        scale = portion_g / 100
        for k in ("calories", "protein_g", "iron_mg", "folate_mcg"):
            out[k] = round(out[k] * scale, 1)
        out["portion_g"] = portion_g
    return out

# --- simple local language helpers ---
LANG_NAMES = {"en":"English","ig":"Igbo","yo":"Yoruba","ha":"Hausa"}

LOCATION_STYLES = {
    "default": {
        "label": "Africa",
        "guide": "simple everyday words that any lay person can understand",
        "tip": "When you can, add familiar foods like beans, egg, fish, meat, green vegetables, or fruit.",
        "tips": [
            "When you can, add familiar foods like beans, egg, fish, meat, green vegetables, or fruit.",
            "A small extra portion of beans, egg, fish, meat, vegetables, or fruit can make the meal stronger.",
            "Try to balance the plate with body-building food and vegetables when they are available.",
        ],
    },
    "nigeria": {
        "label": "Nigeria",
        "guide": "simple Nigerian phrasing, no heavy slang or medical jargon",
        "tip": "For a Nigerian meal, beans, ugu, egg, fish, liver, meat, moi moi, and oranges can help.",
        "tips": [
            "For a Nigerian meal, beans, ugu, egg, fish, liver, meat, moi moi, and oranges can help.",
            "You can make it stronger with beans, moi moi, egg, fish, meat, ugu, or a little fruit.",
            "If it is available, add egg, fish, beans, ugu, liver, or meat to balance the meal better.",
        ],
    },
    "ghana": {
        "label": "Ghana",
        "guide": "simple Ghanaian phrasing with familiar words, no heavy slang or medical jargon",
        "tip": "For a Ghanaian meal, beans, kontomire, egg, fish, meat, groundnuts, and oranges can help.",
        "tips": [
            "For a Ghanaian meal, beans, kontomire, egg, fish, meat, groundnuts, and oranges can help.",
            "You can add kontomire, beans, egg, fish, meat, groundnuts, or fruit to make it more balanced.",
            "If you have some, add fish, egg, beans, kontomire, meat, or groundnuts for a stronger plate.",
        ],
    },
    "kenya": {
        "label": "Kenya",
        "guide": "simple Kenyan phrasing with familiar words, no heavy slang or medical jargon",
        "tip": "For a Kenyan meal, beans, sukuma wiki, egg, fish, meat, milk, and fruit can help.",
        "tips": [
            "For a Kenyan meal, beans, sukuma wiki, egg, fish, meat, milk, and fruit can help.",
            "You can balance it better with beans, sukuma wiki, egg, fish, meat, milk, or fruit.",
            "If possible, add beans, egg, fish, meat, sukuma wiki, or milk to make the meal stronger.",
        ],
    },
    "ethiopia": {
        "label": "Ethiopia",
        "guide": "simple Ethiopian phrasing with familiar words, no heavy slang or medical jargon",
        "tip": "For an Ethiopian meal, shiro, lentils, eggs, meat, fish, greens, and fruit can help.",
        "tips": [
            "For an Ethiopian meal, shiro, lentils, eggs, meat, fish, greens, and fruit can help.",
            "You can make it stronger with shiro, lentils, egg, meat, fish, greens, or fruit.",
            "If it is available, add shiro, lentils, eggs, meat, fish, or greens to balance the meal.",
        ],
    },
    "cameroon": {
        "label": "Cameroon",
        "guide": "simple Cameroonian phrasing with familiar words, no heavy slang or medical jargon",
        "tip": "For a Cameroonian meal, beans, eru, ndole, egg, fish, meat, and fruit can help.",
        "tips": [
            "For a Cameroonian meal, beans, eru, ndole, egg, fish, meat, and fruit can help.",
            "You can add beans, eru, ndole, egg, fish, meat, or fruit to make the meal more complete.",
            "If possible, add fish, egg, beans, eru, ndole, meat, or fruit for a stronger plate.",
        ],
    },
    "west_africa": {
        "label": "West Africa",
        "guide": "simple West African phrasing with familiar words, no heavy slang or medical jargon",
        "tip": "For a West African meal, beans, leafy vegetables, egg, fish, meat, groundnuts, and fruit can help.",
        "tips": [
            "For a West African meal, beans, leafy vegetables, egg, fish, meat, groundnuts, and fruit can help.",
            "You can balance it with beans, egg, fish, meat, leafy vegetables, groundnuts, or fruit.",
            "If you can, add beans, vegetables, egg, fish, meat, or fruit to make the food more complete.",
        ],
    },
}

LOCATION_ALIASES = {
    "nigeria": "nigeria",
    "naija": "nigeria",
    "lagos": "nigeria",
    "abuja": "nigeria",
    "ibadan": "nigeria",
    "kano": "nigeria",
    "port_harcourt": "nigeria",
    "enugu": "nigeria",
    "ghana": "ghana",
    "accra": "ghana",
    "kumasi": "ghana",
    "kenya": "kenya",
    "nairobi": "kenya",
    "mombasa": "kenya",
    "ethiopia": "ethiopia",
    "addis_ababa": "ethiopia",
    "cameroon": "cameroon",
    "douala": "cameroon",
    "yaounde": "cameroon",
    "west_africa": "west_africa",
}


def location_style_key(value):
    token = food_token(value)
    return LOCATION_ALIASES.get(token)


def resolve_location_style(country=None, location=None, food_keys=None):
    for value in (country, location):
        key = location_style_key(value)
        if key:
            return key, LOCATION_STYLES[key]

    for food_key in food_keys or []:
        detection = normalize_food_detection(food_key)
        entry = FOOD_CATALOG.get(detection["key"], {})
        key = location_style_key(detection.get("country")) or location_style_key(entry.get("region"))
        if key:
            return key, LOCATION_STYLES[key]

    return "default", LOCATION_STYLES["default"]


def response_style_payload(style_key, style, lang, source):
    return {
        "country_or_region": style["label"],
        "language": LANG_NAMES.get(lang, "English"),
        "plain_terms": True,
        "local_style": style_key,
        "source": source,
        "varied": True,
    }


def varied_choice(options):
    return random.choice(options)


def style_tip(style):
    return varied_choice(style.get("tips") or [style["tip"]])


def catalog_food_option(key, entry):
    origin, country = catalog_origin(entry)
    return {
        "key": key,
        "name": entry.get("display_name", food_display_name(key)),
        "region": entry.get("region"),
        "country": country,
        "food_origin": origin,
        "aliases": entry.get("aliases", []),
        "mapped": True,
    }


def open_food_option(value):
    key = normalize_food_key(value)
    origin = infer_open_food_origin(key)
    return {
        "key": key,
        "name": food_display_name(key),
        "region": None,
        "country": None,
        "food_origin": origin,
        "aliases": [],
        "mapped": key in FOOD_CATALOG,
    }


def build_plain_summary(total):
    if not any(total.get(k, 0) for k in ("calories", "protein_g", "iron_mg", "folate_mcg")):
        return varied_choice([
            "I found the food, but I do not have enough nutrition details for it yet.",
            "I can see the food, but I do not have enough nutrition information for it yet.",
            "The food was detected, but the nutrition details are not complete yet.",
        ])

    parts = []
    calories = total.get("calories", 0)
    protein = total.get("protein_g", 0)
    iron = total.get("iron_mg", 0)
    folate = total.get("folate_mcg", 0)

    if calories:
        parts.append(varied_choice([
            f"This meal gives about {round(calories)} calories for energy.",
            f"You are getting around {round(calories)} calories from this meal.",
            f"This food gives roughly {round(calories)} calories to fuel the body.",
        ]))
    if protein < 12:
        parts.append(varied_choice([
            "It does not have much body-building food, which is protein.",
            "The protein is on the low side, so the meal needs more body-building food.",
            "It could use more protein to help the body grow and repair itself.",
        ]))
    else:
        parts.append(varied_choice([
            "It has a good amount of body-building food, which is protein.",
            "The protein level looks good for body-building needs.",
            "It gives a useful amount of protein for the body.",
        ]))
    if iron < 4:
        parts.append(varied_choice([
            "It is low in blood-building iron.",
            "The iron is not much, so the meal could support the blood better.",
            "It needs more iron-rich food to help keep the blood strong.",
        ]))
    else:
        parts.append(varied_choice([
            "It has a good amount of blood-building iron.",
            "The iron level looks helpful for strong blood.",
            "It gives a useful amount of iron for the blood.",
        ]))
    if folate < 100:
        parts.append(varied_choice([
            "It needs more folate, a nutrient from foods like green vegetables, beans, and oranges.",
            "The folate is low, so green vegetables, beans, avocado, or oranges would help.",
            "It would be better with more folate-rich food, like vegetables, beans, avocado, or oranges.",
        ]))
    else:
        parts.append(varied_choice([
            "It has a good amount of folate from the food.",
            "The folate level looks helpful.",
            "It gives a useful amount of folate.",
        ]))

    return " ".join(parts)


async def localize_message(text, lang, style, include_tip=False):
    target_language = LANG_NAMES.get(lang, "English")
    if not GEMINI_KEY:
        if style["label"] == "Africa" or not include_tip:
            return text, "rules"
        return f"{text} {style_tip(style)}", "rules"

    prompt = (
        f"Rewrite this nutrition message for a lay person in {style['label']}. "
        f"Use {style['guide']}. Write in {target_language}. "
        "Keep it short, warm, and practical. Avoid medical jargon. "
        "Do not add unsafe advice. Do not change the nutrition meaning. "
        "Vary the phrasing naturally so it does not sound like a repeated template. "
        "Reply with only the rewritten message:\n\n"
        f"{text}"
    )
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{GEMINI_URL}?key={GEMINI_KEY}",
                json={
                    "contents":[{"parts":[{"text":prompt}]}],
                    "generationConfig": {"temperature": 0.85, "topP": 0.9},
                })
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip(), "gemini"
    except Exception:
        if style["label"] == "Africa" or not include_tip:
            return text, "rules"
        return f"{text} {style_tip(style)}", "rules"


# --- simple, clear advice (no MamaBot, instant) ---
IRON_LOW_MESSAGES = [
    "This meal is a bit low in iron. Iron helps keep the blood strong. Add beans, green vegetables, liver, egg, fish, or meat when you can.",
    "Your body may need more iron from this meal. Iron supports strong blood. Beans, leafy vegetables, liver, egg, fish, or meat can help.",
    "This plate could use more blood-building food. Try beans, green vegetables, liver, egg, fish, or meat when they are available.",
]

PROTEIN_LOW_MESSAGES = [
    "This meal does not have much protein. Protein helps the body grow and repair itself. Add egg, fish, beans, meat, milk, or groundnuts when you can.",
    "It could use more body-building food. Egg, fish, beans, meat, milk, or groundnuts can make it stronger.",
    "For a fuller meal, add more protein if you have it. Good choices are egg, fish, beans, meat, milk, or groundnuts.",
]

FOLATE_LOW_MESSAGES = [
    "This meal needs more folate. Folate helps the body and baby grow well. Add green vegetables, beans, avocado, or oranges when you can.",
    "It would be better with more folate-rich food. Green vegetables, beans, avocado, or oranges are helpful options.",
    "Try to add foods that bring folate, like green vegetables, beans, avocado, or oranges, when they are available.",
]

BALANCED_MESSAGES = [
    "This meal looks balanced. Keep adding protein, vegetables, and fruit to your food when you can.",
    "This is a good plate. Keep it up, and try to include body-building food, vegetables, and fruit often.",
    "This meal is doing well. Keep mixing energy food with protein, vegetables, and fruit when available.",
]


def detection_name(detection):
    return food_response(detection, None)["name"]


def build_food_context_advice(detections):
    if not detections:
        return None

    normalized = [normalize_food_detection(item) for item in detections]
    visible = [item for item in normalized if item["key"] != "unknown_food"]
    if not visible:
        return "I am not fully sure what this meal is. Please check the detected food before relying on the advice."

    if len(visible) > 1:
        names = ", ".join(detection_name(item) for item in visible[:4])
        if any(item["food_origin"] == "mixed" for item in visible) or len(visible) > 1:
            return f"This looks like a mixed meal with {names}. Try to balance it with vegetables, fruit, and enough protein during the day."

    item = visible[0]
    name = detection_name(item)
    origin = item["food_origin"]
    country = item.get("country")

    if origin == "african":
        if country:
            return f"This looks like {name} from {country}. Keep the portion sensible and add vegetables, beans, egg, fish, or meat if the meal is light."
        return f"This looks like {name}, an African food. Try to balance it with protein, vegetables, and fruit when you can."
    if origin == "western":
        return f"This looks like {name}. It can be filling, but try adding a side of vegetables or fruit later today."
    if origin == "asian":
        return f"This looks like {name}. It can be a good meal, but try to balance it with vegetables, protein, or fruit later today."
    if origin == "middle_eastern":
        return f"This looks like {name}. It can be satisfying, but add vegetables, fruit, or enough protein during the day."
    if origin == "mixed":
        return f"This looks like {name}, a mixed meal. Try to make sure it has some vegetables and body-building food too."
    if origin == "generic":
        return f"This looks like {name}. Try to balance it with vegetables, fruit, and enough protein during the day."

    return f"This looks like {name}, but I am not fully sure. Please check the detected food, then use the advice as a guide."


def build_advice(total):
    msgs = []
    if total["iron_mg"] < 4:
        msgs.append(varied_choice(IRON_LOW_MESSAGES))
    if total["protein_g"] < 12:
        msgs.append(varied_choice(PROTEIN_LOW_MESSAGES))
    if total["folate_mcg"] < 100:
        msgs.append(varied_choice(FOLATE_LOW_MESSAGES))
    if not msgs:
        return varied_choice(BALANCED_MESSAGES)
    return " ".join(msgs)


def build_coach_advice(total, detections=None):
    context = build_food_context_advice(detections)
    if context:
        if not any(total.get(k, 0) for k in ("calories", "protein_g", "iron_mg", "folate_mcg")):
            return context

        origin = normalize_food_detection(detections[0])["food_origin"] if detections else "unknown"
        if origin in {"western", "asian", "middle_eastern", "mixed", "generic", "unknown"}:
            return context

        return f"{context} {build_advice(total)}"

    return build_advice(total)


MAMABOT_USER_STATUSES = {
    "pregnant": "pregnant mother",
    "pregnancy": "pregnant mother",
    "expecting": "pregnant mother",
    "breastfeeding": "breastfeeding mother",
    "breast_feeding": "breastfeeding mother",
    "lactating": "breastfeeding mother",
    "nursing": "breastfeeding mother",
}


def mamabot_context(*statuses):
    for status in statuses:
        if not status:
            continue

        key = status.strip().lower().replace("-", "_").replace(" ", "_")
        label = MAMABOT_USER_STATUSES.get(key)
        if label:
            return key, label

    return None, None


# --- Gemini plate recognition (pan-African) ---
async def recognize_with_gemini(img_bytes):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_KEY is not configured")

    b64 = base64.b64encode(img_bytes).decode()
    prompt = vision_catalog_prompt()
    async with httpx.AsyncClient(timeout=25) as c:
        r = await c.post(f"{GEMINI_URL}?key={GEMINI_KEY}",
            json={"contents":[{"parts":[{"text":prompt},
                  {"inline_data":{"mime_type":"image/jpeg","data":b64}}]}]})
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return normalize_food_list(_json.loads(txt))

async def maternal_advice(total, user_label, detections=None):
    base = build_coach_advice(total, detections)          # reliable rule-grounded advice
    if not HF_TOKEN:
        return base, "rules"

    try:
        prompt = (f"Rewrite this maternal nutrition advice in warm, simple words for a "
                  f"{user_label}. Keep the same nutrition meaning, but vary the wording "
                  f"naturally so it does not sound like a repeated template:\n\n{base}")
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(MAMA_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 200,
                        "do_sample": True,
                        "temperature": 0.85,
                        "top_p": 0.9,
                    },
                })
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"].replace(prompt, "").strip()
            if text:
                return text, "mamabot"
    except Exception:
        pass
    return base, "rules"


async def advice_for_user(total, *statuses, detections=None):
    _, user_label = mamabot_context(*statuses)
    if user_label:
        return await maternal_advice(total, user_label, detections)

    return build_coach_advice(total, detections), "rules"

# --- endpoints ---
@app.post("/scan-plate")
async def scan_plate(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    country: str | None = Form(None),
    location: str | None = Form(None),
    user_status: str | None = Form(None),
    onboarding_status: str | None = Form(None),
    life_stage: str | None = Form(None),
    maternal_status: str | None = Form(None),
    current_user: dict | None = Depends(get_current_user),
):
    img_bytes = await file.read()
    try:
        detections = await recognize_with_gemini(img_bytes)
    except Exception:
        return {
            "foods": [],
            "summary": "",
            "advice": "",
            "error": "I could not read the plate clearly. Please try again with a clearer food photo.",
            "error_code": "plate_recognition_failed",
        }
    nutritions = await asyncio.gather(*[get_nutrition(n, n.get("portion_g")) for n in detections])
    foods, total = [], {"calories":0,"protein_g":0,"iron_mg":0,"folate_mcg":0}
    for detection, nut in zip(detections, nutritions):
        source = "gemini" if detection["key"] in FOOD_CATALOG else "gemini_open"
        if detection["food_origin"] == "unknown":
            source = "gemini_unknown"
        foods.append(food_response(detection, nut, source=source))
        if nut:
            for k in total: total[k] += nut[k]
    style_key, style = resolve_location_style(country, location, detections)
    summary_en = build_plain_summary(total)
    advice_en, advice_source = await advice_for_user(
        total, user_status, onboarding_status, life_stage, maternal_status,
        detections=detections,
    )
    summary, summary_source = await localize_message(summary_en, lang, style)
    advice, advice_language_source = await localize_message(
        advice_en, lang, style, include_tip=True
    )
    return {
        "foods": foods,
        "total": total,
        "summary": summary,
        "summary_en": summary_en,
        "advice": advice,
        "advice_en": advice_en,
        "advice_source": advice_source,
        "response_style": response_style_payload(
            style_key, style, lang, advice_language_source or summary_source
        ),
    }

@app.post("/scan")   # your own model, single dish
async def scan(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    country: str | None = Form(None),
    location: str | None = Form(None),
    user_status: str | None = Form(None),
    onboarding_status: str | None = Form(None),
    life_stage: str | None = Form(None),
    maternal_status: str | None = Form(None),
    current_user: dict | None = Depends(get_current_user),
):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    with torch.no_grad():
        p = F.softmax(clf(tfm(img).unsqueeze(0)), 1)[0]
    conf, idx = p.max(0)
    model_key = normalize_food_key(classes[int(idx)])
    model_confidence = conf.item()
    selected_detection = normalize_food_detection(model_key)
    recognition_source = "model"
    model_guess = None

    if GEMINI_KEY:
        try:
            gemini_detections = await recognize_with_gemini(img_bytes)
            if gemini_detections:
                gemini_detection = gemini_detections[0]
                gemini_key = gemini_detection["key"]
                gemini_is_unmapped = gemini_key not in FOOD_CATALOG
                gemini_not_in_model = gemini_key not in classes
                should_use_gemini = (
                    gemini_is_unmapped
                    or gemini_not_in_model
                    or gemini_detection["food_origin"] in {"western", "asian", "middle_eastern", "mixed", "unknown"}
                    or model_confidence < MIN_CLASSIFIER_CONFIDENCE
                )

                if should_use_gemini:
                    selected_detection = gemini_detection
                    recognition_source = "gemini_open" if gemini_is_unmapped else "gemini_fallback"
                    if gemini_detection["food_origin"] == "unknown":
                        recognition_source = "gemini_unknown"
                    model_guess = {
                        "key": model_key,
                        "name": food_display_name(model_key),
                        "confidence": round(model_confidence * 100),
                    }
        except Exception:
            pass

    nut = await get_nutrition(selected_detection)
    total = nut if nut else {"calories":0,"protein_g":0,"iron_mg":0,"folate_mcg":0}
    style_key, style = resolve_location_style(country, location, [selected_detection])
    summary_en = build_plain_summary(total)
    advice_en, advice_source = await advice_for_user(
        total, user_status, onboarding_status, life_stage, maternal_status,
        detections=[selected_detection],
    )
    summary, summary_source = await localize_message(summary_en, lang, style)
    advice, advice_language_source = await localize_message(
        advice_en, lang, style, include_tip=True
    )
    confidence = model_confidence if recognition_source == "model" else None
    return {"foods": [food_response(
                selected_detection,
                nut,
                confidence=confidence,
                source=recognition_source,
                model_guess=model_guess,
                needs_review=(
                    recognition_source == "model"
                    and model_confidence < MIN_CLASSIFIER_CONFIDENCE
                ),
            )],
            "summary": summary,
            "summary_en": summary_en,
            "advice": advice,
            "advice_en": advice_en,
            "advice_source": advice_source,
            "response_style": response_style_payload(
                style_key, style, lang, advice_language_source or summary_source
            )}


@app.get("/foods")
def foods(q: str | None = None, limit: int = 50):
    query = food_token(q)
    results = []

    for key, entry in FOOD_CATALOG.items():
        searchable = [key, entry.get("display_name", ""), entry.get("region", "")]
        searchable.extend(entry.get("aliases", []))
        tokens = {food_token(value) for value in searchable if value}

        if query and not any(query in token for token in tokens):
            continue

        results.append(catalog_food_option(key, entry))

    results = results[: max(1, min(limit, 100))]

    if query and all(item["key"] != query for item in results):
        results.append({
            **open_food_option(q),
            "source": "open_food",
        })

    return {
        "foods": results,
        "open_foods_supported": True,
        "correction_source": "backend",
    }


@app.get("/")
def health():
    return {
        "status": "NutriPadi API running",
        "auth_required": AUTH_REQUIRED,
        "supabase_configured": is_supabase_configured(),
    }

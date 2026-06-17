# Deploy to Hugging Face Spaces

You do not need to install Docker on your laptop for Hugging Face Spaces.
Hugging Face reads the `Dockerfile`, builds the container on its own server, and starts the API.

## 1. Add secrets

Open your Space, then go to:

`Settings` -> `Variables and secrets` -> `New secret`

Add these secrets:

- `USDA_API_KEY`
- `HF_TOKEN`
- `GEMINI_KEY`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `AUTH_REQUIRED`

Set `AUTH_REQUIRED` to `true` for production. With that enabled, `/scan` and `/scan-plate` require this header:

`Authorization: Bearer <supabase_access_token>`

## 2. Upload files

Open the **Files** tab in your Hugging Face Space and upload these files from this project:

- `main.py`
- `best.pt`
- `class_names.json`
- `requirements.txt`
- `Dockerfile`
- `README.md`
- `.dockerignore`
- `.gitignore`
- `.env.example`

Do not upload:

- `venv/`
- `__pycache__/`
- `.DS_Store`
- `.env`

Use `.env` only on your computer. For Hugging Face, add the same names and values in **Settings** -> **Variables and secrets**.

## 3. Wait for the build

After upload, Hugging Face will automatically build the Docker image.
Open the **Logs** tab and wait until the app starts.

## 4. Use the API URL

Your API base URL will be:

`https://nnedu-foodscan-backend.hf.space`

Test it in your browser:

`https://nnedu-foodscan-backend.hf.space/`

Use it in React Native:

```ts
const API_BASE_URL = "https://nnedu-foodscan-backend.hf.space";
```

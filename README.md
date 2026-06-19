---
title: Foodscan Backend
emoji: 🍲
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Foodscan Backend

FastAPI backend for food image scanning and maternal nutrition advice.

## Required Hugging Face Secrets

Add these in the Space settings under **Variables and secrets**:

- `USDA_API_KEY`
- `HF_TOKEN`
- `GEMINI_KEY`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `AUTH_REQUIRED`

## Scan advice source

The frontend should send the user's onboarding choice with `/scan` or `/scan-plate`
as one of these optional form fields:

- `user_status`
- `onboarding_status`
- `life_stage`
- `maternal_status`

When the value is `pregnant` or `breastfeeding`, the backend uses MamaBot for
the returned advice. Other values keep the rule-based advice. Responses include
`advice_source` as either `mamabot` or `rules`.

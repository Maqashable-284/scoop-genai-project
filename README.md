# 🤖 Scoop GenAI - სპორტული კვების AI კონსულტანტი

[![Security Grade](https://img.shields.io/badge/Security-B+-green)](CODE_REVIEW_REPORT.md)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## რა არის ეს პროექტი?

ეს არის **ჭკვიანი ჩატბოტი** რომელიც ეხმარება ადამიანებს სპორტული კვების პროდუქტების შერჩევაში. 

წარმოიდგინე რომ მაღაზიაში მიხვალ და გყავს **პირადი კონსულტანტი** რომელიც:
- გახსოვს შენი სახელი და ალერგიები
- იცის ყველა პროდუქტის ფასი და აღწერა
- გირჩევს რა გჭირდება შენი მიზნის მიხედვით

---

## 🧠 როგორ მუშაობს?

```
მომხმარებელი: "გამარჯობა, მე ლუკა მქვია, ლაქტოზის აუტანლობა მაქვს"
     ↓
   [ჩატბოტი ინახავს ამას მონაცემთა ბაზაში]
     ↓
მომხმარებელი: "მირჩიე პროტეინი"
     ↓
   [ჩატბოტს ახსოვს რომ ლაქტოზა არ შეიძლება]
     ↓
ჩატბოტი: "გირჩევ მცენარეულ პროტეინს - Applied Nutrition Critical Plant..."
```

---

## 📁 პროექტის სტრუქტურა

```
scoop-genai-project/
│
├── main.py                    ← მთავარი ფაილი (სერვერი)
├── config.py                  ← პარამეტრები + System Prompt
├── requirements.txt           ← საჭირო ბიბლიოთეკები
├── .env                       ← API keys (საიდუმლო!)
│
├── prompts/
│   └── system_prompt.py       ← AI-ის ინსტრუქციები
│
└── app/
    ├── memory/
    │   └── mongo_store.py     ← მახსოვრობა (MongoDB-ში ინახავს)
    │
    ├── catalog/
    │   └── loader.py          ← პროდუქტების ჩატვირთვა
    │
    └── tools/
        └── user_tools.py      ← Gemini Function Calling Tools
```

---

## 🔧 ტექნოლოგიები

| ტექნოლოგია | ვერსია | რისთვის გამოიყენება |
|------------|--------|---------------------|
| **Google Gemini 2.5 Flash** | Latest | ხელოვნური ინტელექტი |
| **FastAPI** | 0.115+ | Python ვებ-სერვერი |
| **MongoDB** | 7.0+ | მონაცემთა ბაზა |
| **Motor** | 3.6+ | Async MongoDB driver |
| **PyMongo** | 4.10+ | Sync MongoDB (Tools) |
| **slowapi** | 0.1.9 | Rate Limiting |

---

## 🚀 Quick Start

### 1. დააინსტალირე ბიბლიოთეკები:
```bash
pip install -r requirements.txt
```

### 2. შექმენი `.env` ფაილი:
```env
# Required
GEMINI_API_KEY=your_gemini_api_key
MONGODB_URI=mongodb+srv://...
MONGODB_DATABASE=scoop_db

# Security (Production)
ADMIN_TOKEN=your_secure_admin_token
ALLOWED_ORIGINS=https://yourdomain.com
DEBUG=false
```

### 3. გაუშვი სერვერი:
```bash
python3 main.py
```

### 4. შედეგი:
```
INFO: Uvicorn running on http://0.0.0.0:8080
```

---

## 📡 API Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/` | GET | სერვერის სტატუსი | ❌ |
| `/health` | GET | ჯანმრთელობის შემოწმება | ❌ |
| `/chat` | POST | მთავარი ჩატი | ❌ |
| `/chat/stream` | POST | SSE Streaming | ❌ |
| `/sessions` | GET | აქტიური სესიები | ✅ Admin |
| `/session/clear` | POST | სესიის წაშლა | ✅ Admin |

### მაგალითი - Chat:
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "message": "მინდა პროტეინი"}'
```

### მაგალითი - Admin Endpoint:
```bash
curl -X GET http://localhost:8080/sessions \
  -H "X-Admin-Token: your_admin_token"
```

---

## 🔐 Security Features

### Implemented (v1.1.0)

| Feature | Description |
|---------|-------------|
| **Rate Limiting** | 30 requests/minute per IP (slowapi) |
| **Input Validation** | Pydantic validators (user_id, message length) |
| **Admin Authentication** | X-Admin-Token header required |
| **Regex Injection Protection** | `re.escape()` on user input |
| **Error Message Sanitization** | Error IDs instead of stack traces |
| **CORS Warning** | Logs warning if `*` used in production |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | - | Google Gemini API Key |
| `MONGODB_URI` | ✅ | - | MongoDB Connection String |
| `MONGODB_DATABASE` | ❌ | scoop_db | Database Name |
| `ADMIN_TOKEN` | ❌ | - | Admin endpoint auth |
| `ALLOWED_ORIGINS` | ❌ | `*` | CORS origins (comma-separated) |
| `DEBUG` | ❌ | false | Debug mode |

---

## 🧠 მახსოვრობა (Memory)

ჩატბოტს **ახსოვს** ყველაფერი:

| რა ახსოვს | მაგალითი |
|-----------|---------| 
| სახელი | "მე ლუკა მქვია" |
| ალერგიები | "ლაქტოზის აუტანლობა მაქვს" |
| მიზნები | "კუნთის მასა მინდა" |
| წინა კითხვები | "რა გკითხე?" → ახსოვს! |

### როგორ მუშაობს:
1. ყოველი მესიჯი ინახება **MongoDB**-ში
2. როცა ხელახლა წერ, ჩატბოტი ჩატვირთავს ისტორიას
3. **7 დღის** შემდეგ ავტომატურად იშლება (TTL Index)

---

## 🛠️ Gemini Function Calling

ჩატბოტი იყენებს **Automatic Function Calling**-ს:

| Tool | Description |
|------|-------------|
| `get_user_profile` | მომხმარებლის პროფილის წაკითხვა |
| `update_user_profile` | პროფილის განახლება |
| `search_products` | პროდუქტების ძებნა |
| `get_product_details` | დეტალური ინფორმაცია |

---

## 💰 Cost Comparison

| | Claude SDK (ძველი) | Gemini 2.5 Flash |
|--|-------------------|------------------|
| თვეში | ~$1,500 | ~$15 |
| დანაზოგი | - | **99%** |

---

## 📊 Recent Updates

### v1.1.0 (2026-01-13)
- 🔐 **Security Fixes**: 6 P0 vulnerabilities fixed
- 🔧 **Bug Fixes**: Async loop conflict, RepeatedComposite serialization
- ⚡ **Rate Limiting**: slowapi integration
- ✅ **Input Validation**: Pydantic validators

### v1.0.0 (2026-01-12)
- 🚀 Initial release
- 🧠 Gemini 2.5 Flash integration
- 💾 MongoDB persistence
- 🔄 Session management

---

## 📝 Documentation

- [CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md) - Security Audit Results
- [docs/RESPONSE_STYLE_GUIDE.md](docs/RESPONSE_STYLE_GUIDE.md) - AI Response Guidelines

---

## 👥 Authors

შექმნილია **Scoop.ge**-სთვის

**Repository**: https://github.com/Maqashable-284/scoop-genai-project
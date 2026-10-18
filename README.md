# Acme Appliances Ledger Agent

## Files
- `main.py`: FastAPI service
- `requirements.txt`: Python dependencies
- `render.yaml`: Render deployment configuration

## Environment variables
Set these in Render's Environment page:
- `LEDGER_API_URL`: the full private questionData entry URL for your assignment, including the `email`, `quizSign`, `questionId=q-ledger-agent-server`, and root `path=/` parameters.
- `AI_PIPE_TOKEN`: your AI Pipe token. Keep it private; do not commit it to GitHub.
- `AI_PIPE_BASE_URL`: `https://aipipe.org/openrouter/v1` (default in `render.yaml`)
- `AI_MODEL`: `openai/gpt-4.1-nano` (default in `render.yaml`)

## Deploy
1. Create a GitHub repository and upload these three files.
2. In Render, create a new Web Service from that repository.
3. Add the environment variables above in Render. Do not put your quiz URL or AI token directly in source code.
4. Deploy. Visit the service URL ending in `/` to check health. It should return `ready: true` and a positive `orders_loaded`.
5. Test a question with PowerShell:

```powershell
$body = @{ question = "What was the paid revenue by region?" } | ConvertTo-Json
Invoke-RestMethod -Uri "https://YOUR-SERVICE.onrender.com/" -Method Post -ContentType "application/json" -Body $body
```

The response has the required shape: `{"answer": ...}`.

## Notes
The service loads the paginated ledger and currency rates at startup, deduplicates orders using the latest `updated_at`, and uses Asia/Kolkata dates. If the AI model request fails, it uses a small rules-based parser as a fallback.

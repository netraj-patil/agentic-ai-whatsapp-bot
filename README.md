# WhatsApp AI Assistant with Calendar Integration

A professional WhatsApp Business API webhook endpoint powered by an AI assistant that can manage Google Calendar events and perform web searches. Built with FastAPI, LangGraph, and LangChain.

## Features

- **AI-Powered Conversations**: Intelligent responses using Groq's LLaMA 3.3 70B model
- **Google Calendar Integration**: Create, search, update, and delete calendar events via natural language
- **Web Search Capability**: Search the web using Tavily API for real-time information
- **WhatsApp Business API**: Seamless integration with WhatsApp Business Platform
- **Secure Webhooks**: Webhook signature verification for secure communication
- **Background Task Processing**: Efficient message handling with FastAPI background tasks
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Architecture

The application uses a LangGraph-based conversation flow:

```
User Message → Chatbot Node → Tool Router
                    ↓             ↓
              Response Node ← Tool Node
                    ↓
             Final Response
```

### Key Components

- **Agent Service**: Manages AI conversation flow and response generation
- **Graph Nodes**: Implements LangGraph state machine for conversation handling
- **Calendar Tools**: Google Calendar API integration with CRUD operations
- **WhatsApp Routes**: FastAPI endpoints for webhook verification and message handling

## Prerequisites

- Python 3.9+
- Google Cloud Project with Calendar API enabled
- Google Service Account with calendar access
- WhatsApp Business API access (Meta Business Account)
- Groq API key
- Tavily API key

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Copy the example environment file and configure it:
```bash
cp example.env .env
```

Edit `.env` with your credentials (see Configuration section below).

## Configuration

Create a `.env` file at the root level with the following variables:

### Google Calendar Configuration
```env
CALENDAR_ID="your-calendar-id@group.calendar.google.com"
SERVICE_ACCOUNT_FILE_PATH="path/to/your/service-account-file.json"
```

**Setting up Google Calendar:**
1. Create a Google Cloud Project
2. Enable Google Calendar API
3. Create a Service Account and download the JSON key file
4. Share your calendar with the service account email
5. Copy the calendar ID from calendar settings

### WhatsApp Business API Configuration
```env
ACCESS_TOKEN="your-whatsapp-access-token"
APP_ID="your-app-id"
APP_SECRET="your-app-secret"
PHONE_NUMBER_ID="your-phone-number-id"
RECIPIENT_WAID="+919423048340"  # Your WhatsApp number with country code
VERSION="v18.0"
VERIFY_TOKEN="your-webhook-verify-token"
```

**Setting up WhatsApp Business API:**
1. Create a Meta Business Account
2. Set up WhatsApp Business Platform
3. Get your Access Token, App ID, App Secret, and Phone Number ID
4. Set a custom verify token for webhook verification

### AI Service API Keys
```env
GROQ_API_KEY="your-groq-api-key"
TAVILY_API_KEY="your-tavily-api-key"
```

**Getting API Keys:**
- **Groq**: Sign up at [groq.com](https://groq.com) and generate an API key
- **Tavily**: Sign up at [tavily.com](https://tavily.com) and get your API key

## Usage

### Running the Application

**Development mode:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /whatsapp/` - Root endpoint with API information
- `GET /whatsapp/webhook` - Webhook verification endpoint
- `POST /whatsapp/webhook` - Webhook receiver for incoming messages
- `GET /whatsapp/health` - Health check endpoint
- `GET /whatsapp/test` - Test message sending (development only)

### Setting Up WhatsApp Webhook

1. Deploy your application to a public server with HTTPS
2. Go to your Meta App Dashboard
3. Navigate to WhatsApp > Configuration
4. Set your webhook URL: `https://your-domain.com/whatsapp/webhook`
5. Set the verify token (same as in your `.env`)
6. Subscribe to `messages` webhook field

## Example Interactions

**Calendar Operations:**
- "Schedule a meeting tomorrow at 2 PM"
- "What's on my calendar for next week?"
- "Update my meeting on Friday to 3 PM"
- "Delete the dentist appointment"
- "What time is it now?"

**Web Search:**
- "What's the weather like today?"
- "Search for Python best practices"
- "What's the latest news on AI?"

## Project Structure

```
project-root/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── routes/
│   │   └── whatsapp_routes.py  # WhatsApp webhook endpoints
│   └── services/
│       ├── agent.py            # AI agent service
│       ├── graph_nodes.py      # LangGraph conversation nodes
│       └── calendar_tools.py   # Google Calendar tools
├── .env                        # Environment variables (not in git)
├── example.env                 # Example environment file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Logging

Logs are written to both console and file:
- Calendar operations: `calendar_tools.log`
- General application logs: Console output

Log levels can be configured in each module's logging setup.

## Error Handling

The application includes comprehensive error handling:
- Invalid datetime formats
- Missing environment variables
- API call failures
- Webhook signature verification failures
- Tool execution errors

## Security Features

- Webhook signature verification using HMAC-SHA256
- Environment-based configuration (no hardcoded secrets)
- Service account authentication for Google Calendar
- Token-based authentication for WhatsApp API

## Troubleshooting

**Calendar not working:**
- Verify service account has calendar access
- Check `SERVICE_ACCOUNT_FILE_PATH` is correct
- Ensure `CALENDAR_ID` matches your shared calendar

**WhatsApp messages not received:**
- Verify webhook is correctly configured in Meta Dashboard
- Check `VERIFY_TOKEN` matches in both `.env` and Meta Dashboard
- Ensure `ACCESS_TOKEN` and `PHONE_NUMBER_ID` are correct
- Check application logs for errors

**AI responses not working:**
- Verify `GROQ_API_KEY` is valid
- Check API quota limits
- Review logs for specific error messages

## Development

### Testing the Agent

```python
from app.services.agent import generate_response

response = generate_response("What's the current time?")
print(response)
```

### Running Tests

```bash
python -m pytest tests/
```

## Dependencies

Key dependencies:
- `fastapi` - Web framework
- `langchain` - LLM framework
- `langgraph` - Graph-based conversation flow
- `langchain-groq` - Groq LLM integration
- `langchain-tavily` - Web search integration
- `google-api-python-client` - Google Calendar API
- `pydantic` - Data validation

See `requirements.txt` for complete list.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Author

Netraj Patil

## Support

For issues and questions:
- Open an issue on GitHub
- Check the logs for detailed error messages
- Review the Meta WhatsApp Business documentation
- Consult Google Calendar API documentation

## Changelog

### Version 1.0.0
- Initial release
- WhatsApp webhook integration
- Google Calendar CRUD operations
- Web search capability
- LangGraph-based conversation flow

"""ASGI application and command-line entry point for the customer support copilot."""
import uvicorn
from app.config import settings
from app.main import app as application

def main() -> None:
    """Runs the AI Customer Support Copilot web service via uvicorn."""
    print("=" * 65)
    print("  AI Customer Support Copilot - Production-Minded Service")
    print(f"  Listening on: http://{settings.HOST}:{settings.PORT}")
    print(f"  Active Mode : {settings.active_provider_mode()}")
    print(f"  Interactive Docs: http://localhost:{settings.PORT}/docs")
    print("=" * 65)
    uvicorn.run(
        "application:application" if settings.DEBUG else application,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

if __name__ == "__main__":
    main()

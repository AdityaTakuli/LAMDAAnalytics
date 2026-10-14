from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    mappls_api_key: str = Field(..., alias="MAPPLS_API_KEY")
    mappls_client_id: str = Field("", alias="MAPPLS_CLIENT_ID")
    mappls_client_secret: str = Field("", alias="MAPPLS_CLIENT_SECRET")
    serp_api_key: str = Field(..., alias="SERP_API_KEY")
    weather_api_key: str = Field(..., alias="WEATHER_API_KEY")
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field("openai/gpt-oss-120b", alias="GROQ_MODEL")

    weather_provider: str = Field("openweather", alias="WEATHER_PROVIDER")
    http_timeout_seconds: int = Field(30, alias="HTTP_TIMEOUT_SECONDS")
    agent_timeout_seconds: int = Field(90, alias="AGENT_TIMEOUT_SECONDS")
    # Per-agent cap. A hung agent degrades to its defaults instead of running
    # the whole graph into AGENT_TIMEOUT_SECONDS and the mock fallback.
    agent_node_timeout_seconds: float = Field(30.0, alias="AGENT_NODE_TIMEOUT_SECONDS")
    scoring_state_path: str = Field("./data/scoring_state.json", alias="SCORING_STATE_PATH")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    enable_gdelt: bool = Field(False, alias="ENABLE_GDELT")
    enable_comtrade: bool = Field(False, alias="ENABLE_COMTRADE")
    allow_gemini_fallback: bool = Field(True, alias="ALLOW_GEMINI_FALLBACK")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

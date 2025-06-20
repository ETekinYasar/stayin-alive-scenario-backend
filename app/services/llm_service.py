"""
LLM Service - Enhanced AI Integration Service

Bu servis STAYIN ALIVE platformunda gelişmiş AI yetenekleri sağlar:
- Multi-provider LLM management (OpenAI, Anthropic)
- Intelligent prompt template management
- Advanced cost optimization ve budgeting
- Response caching ve performance optimization
- Scenario-specific AI content generation
- Usage analytics ve monitoring
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import asyncio

import structlog
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.redis import redis_client
from app.core.llm_config import llm_config_manager, LLMProvider, ModelConfig
from app.models.user import User
from app.services.admin_config_service import get_admin_config_service
from app.models.system_config import LLMModelConfig
from app.core.database import get_db

logger = structlog.get_logger()


class LLMResponse(BaseModel):
    \"\"\"Standard LLM response model\"\"\"
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    response_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMError(Exception):
    \"\"\"Custom LLM service exceptions\"\"\"
    pass


class RateLimitExceeded(LLMError):
    \"\"\"Rate limit exceeded exception\"\"\"
    pass


class CostLimitExceeded(LLMError):
    \"\"\"Cost limit exceeded exception\"\"\"
    pass


class PromptTemplate(BaseModel):
    \"\"\"Enhanced prompt template model\"\"\"
    id: str
    name: str
    description: str
    template: str
    variables: list[str] = Field(default_factory=list)
    category: str = \"general\"
    scenario_type: str | None = None
    disaster_type: str | None = None
    difficulty_level: str = \"medium\"
    expected_tokens: int = 500
    optimal_temperature: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheConfig(BaseModel):
    \"\"\"Cache configuration for LLM responses\"\"\"
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour default
    max_prompt_length: int = 2000  # Only cache shorter prompts
    exclude_patterns: list[str] = Field(default_factory=lambda: [\"user_specific\", \"timestamp\"])


class ProviderMetrics(BaseModel):
    \"\"\"Provider performance metrics\"\"\"
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    last_error: str | None = None
    last_request: datetime | None = None


class LLMService:
    \"\"\"Core LLM service for AI-powered features\"\"\"
    
    def __init__(self):
        self.config_manager = llm_config_manager
        self._openai_client = None
        self._anthropic_client = None
        self.cache_config = CacheConfig()
        self.metrics: Dict[str, ProviderMetrics] = {}
        
        # Initialize metrics for each provider
        for provider in LLMProvider:
            self.metrics[provider.value] = ProviderMetrics()
        
        # Initialize clients
        asyncio.create_task(self._initialize_clients())
    
    async def _initialize_clients(self):
        \"\"\"Initialize AI provider clients\"\"\"
        try:
            # Initialize OpenAI client with custom HTTP client for proxy support
            if settings.OPENAI_API_KEY:
                import httpx
                http_client = httpx.AsyncClient()
                self._openai_client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    http_client=http_client
                )
                logger.info(\"OpenAI client initialized with custom http_client\")
            
            # Initialize Anthropic client with custom HTTP client
            if hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
                import httpx
                http_client = httpx.AsyncClient()
                self._anthropic_client = AsyncAnthropic(
                    api_key=settings.ANTHROPIC_API_KEY,
                    http_client=http_client
                )
                logger.info(\"Anthropic client initialized with custom http_client\")
            
        except Exception as e:
            logger.error(\"Failed to initialize LLM clients\", error=str(e))
    
    @property
    def openai_client(self) -> AsyncOpenAI:
        \"\"\"Get OpenAI client\"\"\"
        if not self._openai_client:
            raise LLMError(\"OpenAI client not initialized\")
        return self._openai_client
    
    @property
    def anthropic_client(self) -> AsyncAnthropic:
        \"\"\"Get Anthropic client\"\"\"
        if not self._anthropic_client:
            raise LLMError(\"Anthropic client not initialized\")
        return self._anthropic_client
    
    async def get_model_for_task_admin(self, task_name: str) -> Optional[LLMModelConfig]:
        \"\"\"Get model for task from admin configuration\"\"\"
        try:
            from app.core.database import get_db
            db = next(get_db())
            admin_service = get_admin_config_service(db)
            return await admin_service.get_model_for_task(task_name)
        except Exception as e:
            logger.error(\"Failed to get admin model configuration\", error=str(e), task_name=task_name)
            return None
    
    async def generate_for_task(
        self,
        task_name: str,
        prompt: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        \"\"\"
        Generate completion optimized for specific task
        
        This method automatically selects the best model for the task,
        applies task-specific configurations, and handles structured output
        \"\"\"
        # First try to get model from admin configuration
        logger.info(\"🔍 Starting model selection for task\", task_name=task_name, user_id=user_id)
        
        admin_model = await self.get_model_for_task_admin(task_name)
        logger.info(\"🔍 Admin model lookup result\", 
                   admin_model_found=admin_model is not None,
                   admin_model_id=admin_model.model_id if admin_model else None,
                   admin_model_active=admin_model.is_active if admin_model else None)
        
        if admin_model and admin_model.is_active:
            # Use admin configured model
            model_id = admin_model.model_id
            provider_model_id = admin_model.provider_model_id
            logger.info(
                \"✅ Using admin configured model for task\",
                task_name=task_name,
                selected_model=model_id,
                provider_model=provider_model_id,
                user_id=user_id
            )
        else:
            # Fallback to static configuration
            model_config = self.config_manager.get_model_for_task(task_name)
            model_id = model_config.id
            provider_model_id = model_config.provider_model_id
            logger.info(
                \"⚠️ Falling back to static configuration\",
                task_name=task_name,
                selected_model=model_id,
                provider_model=provider_model_id,
                reason=\"Admin model not found or inactive\"
            )
        
        # Generate using selected model
        return await self.generate_completion(
            prompt=prompt,
            model=model_id,
            user_id=user_id,
            metadata={\"task_name\": task_name, \"context\": context},
            **kwargs
        )
    
    async def generate_completion(
        self,
        prompt: str,
        model: str = \"gpt-4.1-mini\",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> LLMResponse:
        \"\"\"Generate text completion using specified model\"\"\"
        
        start_time = datetime.now()
        cache_key = None
        
        try:
            # Check cache if enabled
            if use_cache and self.cache_config.enabled and len(prompt) <= self.cache_config.max_prompt_length:
                cache_key = self._generate_cache_key(prompt, model, temperature, max_tokens)
                cached_response = await self._get_cached_response(cache_key)
                if cached_response:
                    logger.info(\"Cache hit for LLM request\", model=model, user_id=user_id)
                    return cached_response
            
            # Get model configuration
            model_config = self.config_manager.get_model(model)
            if not model_config:
                raise LLMError(f\"Model {model} not found in configuration\")
            
            # Check rate limits and costs
            await self._check_rate_limits(user_id, model)
            await self._check_cost_limits(user_id, model_config, max_tokens or 1000)
            
            # Generate completion based on provider
            if model_config.provider == LLMProvider.OPENAI:
                response = await self._generate_openai_completion(
                    prompt, model_config, max_tokens, temperature
                )
            elif model_config.provider == LLMProvider.ANTHROPIC:
                response = await self._generate_anthropic_completion(
                    prompt, model_config, max_tokens, temperature, None
                )
            else:
                raise LLMError(f\"Unsupported provider: {model_config.provider}\")
            
            # Calculate metrics
            end_time = datetime.now()
            response.response_time = (end_time - start_time).total_seconds()
            response.cost_estimate = self._calculate_cost(model_config, response.tokens_used)
            
            # Add metadata
            if metadata:
                response.metadata.update(metadata)
            
            # Update metrics
            await self._update_metrics(model_config.provider, response, user_id)
            
            # Cache response if enabled
            if use_cache and cache_key and self.cache_config.enabled:
                await self._cache_response(cache_key, response)
            
            logger.info(
                \"LLM completion generated successfully\",
                model=model,
                tokens=response.tokens_used,
                cost=response.cost_estimate,
                user_id=user_id
            )
            
            return response
            
        except Exception as e:
            # Update error metrics
            provider = self.config_manager.get_model(model).provider if self.config_manager.get_model(model) else \"unknown\"
            await self._update_error_metrics(provider, str(e))
            
            logger.error(
                \"LLM completion failed\",
                error=str(e),
                model=model,
                user_id=user_id
            )
            raise LLMError(f\"Completion generation failed: {str(e)}\")
    
    async def generate_structured_response(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: str = \"gpt-4.1-mini\",
        context: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        \"\"\"
        Generate structured JSON response matching the provided schema
        
        Args:
            prompt: User prompt
            schema: JSON schema for response validation
            model: Model to use
            context: Additional context
            system_message: System instruction
            temperature: Creativity level
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON response matching the schema
        \"\"\"
        start_time = datetime.now()
        
        try:
            # Check if API is available
            model_config = self.config_manager.get_model(model)
            if not model_config:
                raise LLMError(f\"Model {model} not found in configuration\")
            
            # Check if model supports structured output natively
            from app.core.llm_config import ModelCapability
            if (ModelCapability.STRUCTURED_OUTPUT in model_config.capabilities and 
                model_config.provider == LLMProvider.OPENAI and 
                self._openai_client):
                
                # Use native structured output
                response = await self._generate_openai_structured_output(
                    prompt, schema, model_config, max_tokens, temperature, system_message
                )
            else:
                # Fallback to prompt engineering
                structured_prompt = self._create_structured_prompt(prompt, schema, context, system_message)
                response = await self.generate_completion(
                    structured_prompt, model, max_tokens, temperature, user_id
                )
            
            # Parse and validate JSON response
            try:
                parsed_response = json.loads(response.content)
                # TODO: Add schema validation here
                return parsed_response
            except json.JSONDecodeError as e:
                logger.warning(\"Failed to parse structured response as JSON\", error=str(e), content=response.content[:200])
                # Try to extract JSON from response
                cleaned_response = self._extract_json_from_text(response.content)
                return json.loads(cleaned_response)
                
        except Exception as e:
            logger.error(\"Structured response generation failed\", error=str(e), model=model, user_id=user_id)
            
            # Return fallback response
            return {
                \"error\": True,
                \"message\": \"Failed to generate structured response\",
                \"fallback_used\": True,
                \"source\": \"fallback\",
                \"tokens_used\": 0,
                \"ai_model_used\": model
            }
    
    async def _generate_openai_completion(
        self,
        prompt: str,
        model_config: 'ModelConfig',
        max_tokens: Optional[int],
        temperature: float
    ) -> LLMResponse:
        \"\"\"Generate completion using OpenAI API\"\"\"
        
        # Prepare messages
        messages = []
        messages.append({\"role\": \"user\", \"content\": prompt})
        
        response = await self.openai_client.chat.completions.create(
            model=model_config.provider_model_id,
            messages=messages,
            max_tokens=max_tokens or model_config.max_tokens,
            temperature=temperature
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_config.id,
            tokens_used=response.usage.total_tokens,
            cost_estimate=0.0,  # Will be calculated later
            response_time=0.0   # Will be calculated later
        )
    
    async def _generate_anthropic_completion(
        self,
        prompt: str,
        model_config: 'ModelConfig',
        max_tokens: Optional[int],
        temperature: float,
        system_message: Optional[str]
    ) -> LLMResponse:
        \"\"\"Generate completion using Anthropic API\"\"\"
        
        # Anthropic uses a different message format
        messages = [{\"role\": \"user\", \"content\": prompt}]
        
        response = await self.anthropic_client.messages.create(
            model=model_config.provider_model_id,
            max_tokens=max_tokens or model_config.max_tokens,
            temperature=temperature,
            system=system_message or \"\",
            messages=messages
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=model_config.id,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            cost_estimate=0.0,
            response_time=0.0
        )
    
    async def _stream_openai_completion(
        self,
        prompt: str,
        model_config: 'ModelConfig',
        max_tokens: Optional[int],
        temperature: float
    ) -> AsyncGenerator[str, None]:
        \"\"\"Stream completion using OpenAI API\"\"\"
        
        messages = [{\"role\": \"user\", \"content\": prompt}]
        
        stream = await self.openai_client.chat.completions.create(
            model=model_config.provider_model_id,
            messages=messages,
            max_tokens=max_tokens or model_config.max_tokens,
            temperature=temperature,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def _generate_cache_key(self, prompt: str, model: str, temperature: float, max_tokens: Optional[int]) -> str:
        \"\"\"Generate cache key for request\"\"\"
        key_data = f\"{prompt}:{model}:{temperature}:{max_tokens}\"
        return f\"llm_cache:{hashlib.md5(key_data.encode()).hexdigest()}\"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        \"\"\"Get cached response if available\"\"\"
        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                return LLMResponse.parse_raw(cached_data)
        except Exception as e:
            logger.warning(\"Cache retrieval failed\", error=str(e))
        return None
    
    async def _cache_response(self, cache_key: str, response: LLMResponse):
        \"\"\"Cache response for future use\"\"\"
        try:
            await redis_client.set(
                cache_key, 
                response.json(), 
                ex=self.cache_config.ttl_seconds
            )
        except Exception as e:
            logger.warning(\"Cache storage failed\", error=str(e))
    
    async def _check_rate_limits(self, user_id: Optional[str], model: str):
        \"\"\"Check rate limits for user and model\"\"\"
        if not user_id:
            return
        
        # Implement rate limiting logic here
        # For now, just return
        pass
    
    async def _check_cost_limits(self, user_id: Optional[str], model_config: 'ModelConfig', estimated_tokens: int):
        \"\"\"Check cost limits for user\"\"\"
        if not user_id:
            return
        
        # Implement cost checking logic here
        # For now, just return
        pass
    
    def _calculate_cost(self, model_config: 'ModelConfig', tokens_used: int) -> float:
        \"\"\"Calculate cost for token usage\"\"\"
        # This is a simplified calculation
        return (tokens_used / 1000) * model_config.cost_per_1k_tokens
    
    async def _update_metrics(self, provider: LLMProvider, response: LLMResponse, user_id: Optional[str]):
        \"\"\"Update provider metrics\"\"\"
        metrics = self.metrics[provider.value]
        metrics.total_requests += 1
        metrics.total_tokens += response.tokens_used
        metrics.total_cost += response.cost_estimate
        metrics.last_request = datetime.now()
        
        # Update average response time
        if metrics.total_requests == 1:
            metrics.avg_response_time = response.response_time
        else:
            metrics.avg_response_time = (
                (metrics.avg_response_time * (metrics.total_requests - 1) + response.response_time) 
                / metrics.total_requests
            )
    
    async def _update_error_metrics(self, provider: str, error_message: str):
        \"\"\"Update error metrics for provider\"\"\"
        if provider in self.metrics:
            metrics = self.metrics[provider]
            metrics.last_error = error_message
            # Update error rate calculation would go here
    
    def _create_structured_prompt(
        self, 
        prompt: str, 
        schema: Dict[str, Any], 
        context: Optional[str], 
        system_message: Optional[str]
    ) -> str:
        \"\"\"Create structured prompt for JSON response\"\"\"
        
        schema_str = json.dumps(schema, indent=2)
        
        structured_prompt = f\"\"\"
{system_message or 'You are a helpful AI assistant.'}

{f'Context: {context}' if context else ''}

User Request: {prompt}

Please respond with a valid JSON object that matches this exact schema:
{schema_str}

Important:
- Return ONLY the JSON object, no additional text
- Ensure all required fields are present
- Use appropriate data types as specified in the schema
- Do not include any markdown formatting or code blocks

JSON Response:
\"\"\"
        
        return structured_prompt.strip()
    
    def _extract_json_from_text(self, text: str) -> str:
        \"\"\"Extract JSON from text response\"\"\"
        # Simple extraction - look for { } blocks
        import re
        json_match = re.search(r'\\{.*\\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return text
    
    async def _generate_openai_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model_config: 'ModelConfig',
        max_tokens: Optional[int],
        temperature: float,
        system_message: Optional[str]
    ) -> LLMResponse:
        \"\"\"Generate structured output using OpenAI's native support\"\"\"
        
        messages = []
        if system_message:
            messages.append({\"role\": \"system\", \"content\": system_message})
        messages.append({\"role\": \"user\", \"content\": prompt})
        
        # Note: This would use OpenAI's structured output feature when available
        # For now, fallback to regular completion
        response = await self.openai_client.chat.completions.create(
            model=model_config.provider_model_id,
            messages=messages,
            max_tokens=max_tokens or model_config.max_tokens,
            temperature=temperature
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_config.id,
            tokens_used=response.usage.total_tokens,
            cost_estimate=0.0,
            response_time=0.0
        )


# Global service instance
llm_service = LLMService()


# Service factory function
def get_llm_service() -> LLMService:
    \"\"\"Get LLM service instance\"\"\"
    return llm_service"
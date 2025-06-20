"""
AI Scenarios API - Intelligent Scenario Generation Endpoints

Bu API kullanıcılara AI ile kişiselleştirilmiş senaryo üretimi sağlar
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db
from app.models.user import User
from app.services.ai_scenario_service import (
    ai_scenario_service,
    ScenarioRequirements,
    GeneratedScenario,
    ScenarioNode
)
from app.services.ai_profile_service import UserProfileContext, FamilyMember, PetInfo

logger = structlog.get_logger()

router = APIRouter()


# Enhanced request/response models (placed before endpoints)

class EnhancedScenarioRequest(BaseModel):
    \"\"\"Enhanced scenario generation request\"\"\"
    scenario_type: str = Field(..., description=\"Type of scenario (natural_disaster, tech_failure, health_crisis)\")
    disaster_type: str = Field(..., description=\"Specific disaster type (earthquake, flood, fire, etc.)\")
    difficulty_level: str = Field(default=\"medium\", description=\"Difficulty level (easy, medium, hard)\")
    user_profile: dict[str, Any] = Field(..., description=\"User profile information\")
    location_context: dict[str, Any] | None = Field(None, description=\"Location context from geo service\")
    custom_requirements: dict[str, Any] | None = Field(None, description=\"Custom scenario requirements\")


class ProviderOptimizationRequest(BaseModel):
    \"\"\"Provider optimization request\"\"\"
    prompt_type: str = Field(..., description=\"Type of prompt (scenario_generation, content_creation, etc.)\")
    budget_constraint: float | None = Field(None, description=\"Budget constraint in USD\")
    speed_priority: bool = Field(default=False, description=\"Prioritize speed over quality\")
    quality_priority: bool = Field(default=False, description=\"Prioritize quality over cost\")


class CacheAnalyticsResponse(BaseModel):
    \"\"\"Cache analytics response\"\"\"
    cache_hit_rate: float
    total_requests: int
    cache_hits: int
    cache_misses: int
    average_response_time: float
    total_cost_saved: float


class ProviderPerformanceResponse(BaseModel):
    \"\"\"Provider performance response\"\"\"
    provider: str
    total_requests: int
    average_response_time: float
    total_cost: float
    error_rate: float
    last_error: str | None


# Core request/response models

class ScenarioGenerationRequest(BaseModel):
    \"\"\"Senaryo üretim isteği\"\"\"
    profile_context: UserProfileContext
    requirements: ScenarioRequirements


class QuickScenarioRequest(BaseModel):
    \"\"\"Hızlı senaryo üretim isteği\"\"\"
    disaster_type: str = Field(..., description=\"Felaket türü (earthquake, fire, flood, etc.)\")
    difficulty_level: str = Field(default=\"medium\", description=\"Zorluk seviyesi (easy, medium, hard)\")
    duration_preference: str = Field(default=\"medium\", description=\"Süre tercihi (short, medium, long)\")
    location: str | None = Field(None, description=\"Konum bilgisi\")


class ScenarioCustomizationRequest(BaseModel):
    \"\"\"Senaryo özelleştirme isteği\"\"\"
    scenario_id: str
    customization_type: str = Field(..., description=\"Type of customization (difficulty, content, duration)\")
    target_value: str = Field(..., description=\"Target value for customization\")
    preserve_structure: bool = Field(default=True, description=\"Whether to preserve scenario structure\")


class ScenarioEnhancementRequest(BaseModel):
    \"\"\"Senaryo geliştirme isteği\"\"\"
    scenario_data: Dict[str, Any] = Field(..., description=\"Existing scenario data\")
    enhancement_type: str = Field(..., description=\"Type of enhancement (realism, difficulty, educational_value)\")
    profile_context: UserProfileContext
    focus_areas: List[str] = Field(default_factory=list, description=\"Specific areas to focus on\")


class ScenarioVariationRequest(BaseModel):
    \"\"\"Senaryo varyasyon isteği\"\"\"
    base_scenario: Dict[str, Any] = Field(..., description=\"Base scenario to create variations from\")
    variation_count: int = Field(default=3, ge=1, le=5, description=\"Number of variations to create\")
    variation_types: List[str] = Field(default_factory=list, description=\"Types of variations (location, timing, complexity)\")


class ScenarioQualityRequest(BaseModel):
    \"\"\"Senaryo kalite analizi isteği\"\"\"
    scenario_data: Dict[str, Any] = Field(..., description=\"Scenario data to analyze\")
    evaluation_criteria: List[str] = Field(default_factory=list, description=\"Specific criteria to evaluate\")
    target_audience: str = Field(default=\"general\", description=\"Target audience (beginner, intermediate, advanced)\")


# Response models

class ScenarioGenerationResponse(BaseModel):
    \"\"\"Senaryo üretim yanıtı\"\"\"
    scenario: GeneratedScenario
    generation_time: float
    ai_model_used: str
    tokens_used: int
    personalization_score: float


class ScenarioCustomizationResponse(BaseModel):
    \"\"\"Senaryo özelleştirme yanıtı\"\"\"
    customized_scenario: Dict[str, Any]
    changes_applied: List[str]
    customization_score: float
    recommendations: List[str]


class ScenarioEnhancementResponse(BaseModel):
    \"\"\"Senaryo geliştirme yanıtı\"\"\"
    enhanced_scenario: Dict[str, Any]
    changes: List[Dict[str, Any]]
    improvement_summary: str
    enhancement_type: str


class ScenarioVariationResponse(BaseModel):
    \"\"\"Senaryo varyasyon yanıtı\"\"\"
    variations: List[Dict[str, Any]]
    base_scenario_title: str
    variation_count: int


class ScenarioQualityResponse(BaseModel):
    \"\"\"Senaryo kalite analizi yanıtı\"\"\"
    overall_score: float
    criteria_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    priority_areas: List[str]


# API Endpoints

@router.post(\"/generate\", response_model=ScenarioGenerationResponse)
async def generate_personalized_scenario(
    request: ScenarioGenerationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    Kullanıcı profiline göre kişiselleştirilmiş senaryo üret
    
    Bu endpoint:
    - Kullanıcının konum, aile yapısı ve deneyim seviyesini dikkate alır
    - Seçilen felaket türü ve zorluğa göre özelleştirir
    - Gerçekçi başlangıç koşulları ve dinamik dallanma yapısı oluşturur
    - Eğitim hedefleri ve başarı kriterlerini belirler
    \"\"\"
    try:
        start_time = datetime.now()
        
        # AI ile kişiselleştirilmiş senaryo üret
        scenario = await ai_scenario_service.generate_personalized_scenario(
            user_id=str(current_user.id),
            profile_context=request.profile_context,
            requirements=request.requirements
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Personalization score hesapla
        personalization_score = calculate_personalization_score(
            request.profile_context, 
            request.requirements
        )
        
        logger.info(
            \"Personalized scenario generated\",
            user_id=str(current_user.id),
            disaster_type=request.requirements.disaster_type,
            difficulty=request.requirements.difficulty_level,
            nodes_count=len(scenario.nodes),
            generation_time=generation_time,
            personalization_score=personalization_score
        )
        
        # Admin config'den kullanılan model'i al
        try:
            from app.services.admin_config_service import get_admin_config_service
            from app.core.database import get_db
            admin_db = next(get_db())
            admin_service = get_admin_config_service(admin_db)
            admin_model = await admin_service.get_model_for_task(\"scenario_generation\")
            model_used = admin_model.model_id if admin_model and admin_model.is_active else \"gpt-4.1-mini\"
        except Exception:
            model_used = \"gpt-4.1-mini\"  # Fallback
        
        return ScenarioGenerationResponse(
            scenario=scenario,
            generation_time=generation_time,
            ai_model_used=model_used,
            tokens_used=0,  # TODO: Get from LLM response
            personalization_score=personalization_score
        )
        
    except Exception as e:
        logger.error(
            \"Scenario generation failed\",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Scenario generation failed: {str(e)}\"
        )


@router.post(\"/quick-generate\", response_model=ScenarioGenerationResponse)
async def generate_quick_scenario(
    request: QuickScenarioRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    Hızlı senaryo üretimi - minimal profil bilgisi ile
    
    Bu endpoint daha az kişiselleştirme ile hızlı senaryo üretir
    \"\"\"
    try:
        start_time = datetime.now()
        
        # Minimal context oluştur
        minimal_context = UserProfileContext(
            location=request.location,
            family_members=[],
            pets=[],
            current_inventory={},
            experience_level=\"beginner\"
        )
        
        minimal_requirements = ScenarioRequirements(
            disaster_type=request.disaster_type,
            difficulty_level=request.difficulty_level,
            duration_preference=request.duration_preference,
            focus_areas=[\"basic_safety\"]
        )
        
        scenario = await ai_scenario_service.generate_personalized_scenario(
            user_id=str(current_user.id),
            profile_context=minimal_context,
            requirements=minimal_requirements
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            \"Quick scenario generated\",
            user_id=str(current_user.id),
            disaster_type=request.disaster_type,
            generation_time=generation_time
        )
        
        # Admin config'den kullanılan model'i al  
        try:
            from app.services.admin_config_service import get_admin_config_service
            from app.core.database import get_db
            admin_db = next(get_db())
            admin_service = get_admin_config_service(admin_db)
            admin_model = await admin_service.get_model_for_task(\"scenario_generation\")
            model_used = admin_model.model_id if admin_model and admin_model.is_active else \"gpt-4.1-mini\"
        except Exception:
            model_used = \"gpt-4.1-mini\"  # Fallback
        
        return ScenarioGenerationResponse(
            scenario=scenario,
            generation_time=generation_time,
            ai_model_used=model_used,
            tokens_used=0,
            personalization_score=0.3  # Low due to minimal profile
        )
        
    except Exception as e:
        logger.error(\"Quick scenario generation failed\", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Quick scenario generation failed: {str(e)}\"
        )


@router.post(\"/customize\", response_model=ScenarioCustomizationResponse)
async def customize_scenario(
    request: ScenarioCustomizationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    Mevcut senaryoyu özelleştir
    
    Bu endpoint var olan bir senaryoyu belirli kriterlere göre özelleştirir
    \"\"\"
    try:
        customized_scenario = await ai_scenario_service.customize_scenario(
            user_id=str(current_user.id),
            scenario_id=request.scenario_id,
            customization_type=request.customization_type,
            target_value=request.target_value,
            preserve_structure=request.preserve_structure
        )
        
        logger.info(
            \"Scenario customized\",
            user_id=str(current_user.id),
            scenario_id=request.scenario_id,
            customization_type=request.customization_type
        )
        
        return ScenarioCustomizationResponse(
            customized_scenario=customized_scenario[\"scenario\"],
            changes_applied=customized_scenario[\"changes\"],
            customization_score=customized_scenario[\"score\"],
            recommendations=customized_scenario[\"recommendations\"]
        )
        
    except Exception as e:
        logger.error(\"Scenario customization failed\", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Scenario customization failed: {str(e)}\"
        )


@router.post(\"/enhance\", response_model=ScenarioEnhancementResponse)
async def enhance_scenario(
    request: ScenarioEnhancementRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    Senaryoyu geliştir ve iyileştir
    
    Bu endpoint var olan bir senaryoyu daha gerçekçi ve eğitici hale getirir
    \"\"\"
    try:
        enhanced_data = await ai_scenario_service.enhance_existing_scenario(
            user_id=str(current_user.id),
            scenario_data=request.scenario_data,
            profile_context=request.profile_context,
            enhancement_type=request.enhancement_type
        )
        
        logger.info(
            \"Scenario enhanced\",
            user_id=str(current_user.id),
            enhancement_type=request.enhancement_type
        )
        
        return ScenarioEnhancementResponse(
            enhanced_scenario=enhanced_data[\"scenario\"],
            changes=enhanced_data[\"changes\"],
            improvement_summary=enhanced_data[\"summary\"],
            enhancement_type=request.enhancement_type
        )
        
    except Exception as e:
        logger.error(\"Scenario enhancement failed\", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Scenario enhancement failed: {str(e)}\"
        )


@router.post(\"/variations\", response_model=ScenarioVariationResponse)
async def create_scenario_variations(
    request: ScenarioVariationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    Senaryo varyasyonları oluştur
    
    Bu endpoint bir base scenaryodan farklı varyasyonlar üretir
    \"\"\"
    try:
        variations = await ai_scenario_service.create_scenario_variations(
            user_id=str(current_user.id),
            base_scenario=request.base_scenario,
            variation_count=request.variation_count,
            variation_types=request.variation_types
        )
        
        logger.info(
            \"Scenario variations created\",
            user_id=str(current_user.id),
            variation_count=len(variations)
        )
        
        return ScenarioVariationResponse(
            variations=variations,
            base_scenario_title=request.base_scenario.get(\"title\", \"Unknown\"),
            variation_count=len(variations)
        )
        
    except Exception as e:
        logger.error(\"Scenario variation creation failed\", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Scenario variation creation failed: {str(e)}\"
        )


@router.post(\"/analyze-quality\", response_model=ScenarioQualityResponse)
async def analyze_scenario_quality(
    request: ScenarioQualityRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    Senaryo kalitesini analiz et
    
    Bu endpoint bir senaryonun kalitesini değerlendirir ve iyileştirme önerileri sunar
    \"\"\"
    try:
        quality_analysis = await ai_scenario_service.analyze_scenario_quality(
            user_id=str(current_user.id),
            scenario_data=request.scenario_data,
            evaluation_criteria=request.evaluation_criteria,
            target_audience=request.target_audience
        )
        
        logger.info(
            \"Scenario quality analyzed\",
            user_id=str(current_user.id),
            overall_score=quality_analysis[\"overall_score\"]
        )
        
        return ScenarioQualityResponse(
            overall_score=quality_analysis[\"overall_score\"],
            criteria_scores=quality_analysis[\"criteria_scores\"],
            strengths=quality_analysis[\"strengths\"],
            weaknesses=quality_analysis[\"weaknesses\"],
            improvement_suggestions=quality_analysis[\"suggestions\"],
            priority_areas=quality_analysis[\"priority_areas\"]
        )
        
    except Exception as e:
        logger.error(\"Scenario quality analysis failed\", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Scenario quality analysis failed: {str(e)}\"
        )


@router.get(\"/cache-analytics\", response_model=CacheAnalyticsResponse)
async def get_cache_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    Cache analytics ve performance metrics
    \"\"\"
    try:
        analytics = await ai_scenario_service.get_cache_analytics(
            user_id=str(current_user.id)
        )
        
        return CacheAnalyticsResponse(**analytics)
        
    except Exception as e:
        logger.error(\"Cache analytics retrieval failed\", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Cache analytics retrieval failed: {str(e)}\"
        )


@router.get(\"/provider-performance\", response_model=List[ProviderPerformanceResponse])
async def get_provider_performance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    LLM provider performance metrics
    \"\"\"
    try:
        performance_data = await ai_scenario_service.get_provider_performance(
            user_id=str(current_user.id)
        )
        
        return [ProviderPerformanceResponse(**data) for data in performance_data]
        
    except Exception as e:
        logger.error(\"Provider performance retrieval failed\", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Provider performance retrieval failed: {str(e)}\"
        )


@router.post(\"/optimize-provider\")
async def optimize_provider_selection(
    request: ProviderOptimizationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    \"\"\"
    Provider selection optimization based on requirements
    \"\"\"
    try:
        optimization_result = await ai_scenario_service.optimize_provider_selection(
            user_id=str(current_user.id),
            prompt_type=request.prompt_type,
            budget_constraint=request.budget_constraint,
            speed_priority=request.speed_priority,
            quality_priority=request.quality_priority
        )
        
        logger.info(
            \"Provider optimization completed\",
            user_id=str(current_user.id),
            recommended_provider=optimization_result[\"recommended_provider\"]
        )
        
        return optimization_result
        
    except Exception as e:
        logger.error(\"Provider optimization failed\", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f\"Provider optimization failed: {str(e)}\"
        )


# Helper functions

def calculate_personalization_score(
    profile_context: UserProfileContext, 
    requirements: ScenarioRequirements
) -> float:
    \"\"\"
    Kişiselleştirme skoru hesapla
    \"\"\"
    score = 0.0
    
    # Location context +0.2
    if profile_context.location:
        score += 0.2
    
    # Family members +0.2
    if profile_context.family_members:
        score += 0.2
    
    # Housing details +0.15
    if profile_context.housing_details:
        score += 0.15
    
    # Inventory +0.15
    if profile_context.current_inventory:
        score += 0.15
    
    # Experience level +0.1
    if profile_context.experience_level:
        score += 0.1
    
    # Budget range +0.1
    if profile_context.budget_range:
        score += 0.1
    
    # Requirements detail +0.1
    if len(requirements.focus_areas) > 2:
        score += 0.1
    
    return min(score, 1.0)
"
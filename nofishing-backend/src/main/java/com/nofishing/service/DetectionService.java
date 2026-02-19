package com.nofishing.service;

import com.nofishing.client.MlServiceClient;
import com.nofishing.dto.DetectionRequest;
import com.nofishing.dto.DetectionResponse;
import com.nofishing.dto.MlServiceResponse;
import com.nofishing.entity.DetectionHistory;
import com.nofishing.exception.MlServiceException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * Core Detection Service
 *
 * Orchestrates the phishing detection workflow:
 * 1. Check cache for existing result
 * 2. If not cached, call ML service
 * 3. Cache the result for future requests
 *
 * @author NoFishing Team
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DetectionService {

    private final MlServiceClient mlServiceClient;
    private final UrlFeatureExtractor featureExtractor;
    private final UrlCacheService cacheService;
    private final DetectionHistoryService historyService;
    private final WhitelistService whitelistService;
    private final BlacklistService blacklistService;

    @Value("${nofishing.detection.threshold:0.5}")
    private double detectionThreshold;

    /**
     * Detect if a URL is phishing
     *
     * @param request Detection request containing URL and options
     * @return Detection response with classification result
     */
    public DetectionResponse detectUrl(DetectionRequest request) {
        long startTime = System.currentTimeMillis();
        String url = request.getUrl();

        log.info("Starting detection for URL: {}", url);

        // Check whitelist first - whitelisted URLs are always safe
        if (whitelistService.isWhitelisted(url)) {
            log.info("URL is whitelisted: {}", url);
            return buildWhitelistResponse(url);
        }

        // Check blacklist - blacklisted URLs are always phishing
        if (blacklistService.isBlacklisted(url)) {
            log.info("URL is blacklisted: {}", url);
            return buildBlacklistResponse(url);
        }

        // Check cache first
        DetectionResponse cachedResponse = cacheService.getCachedResult(url);
        if (cachedResponse != null) {
            log.info("Cache HIT for URL: {}", url);
            cachedResponse.setFromCache(true);
            return cachedResponse;
        }

        // Extract URL features
        Map<String, Object> features = featureExtractor.extractFeatures(url);
        Double heuristicScore = (Double) features.get("heuristicScore");

        // Call ML service for classification
        DetectionResponse response;
        try {
            MlServiceResponse mlResponse = mlServiceClient.classifyUrl(
                    url,
                    Boolean.TRUE.equals(request.getFetchContent())
            );

            response = buildDetectionResponse(url, mlResponse, features);
            response.setFromCache(false);

        } catch (MlServiceException e) {
            log.warn("ML service unavailable, falling back to heuristic analysis: {}", e.getMessage());

            // Fallback to heuristic analysis only
            response = buildHeuristicResponse(url, heuristicScore, features);
        }

        // Update processing time
        response.setProcessingTimeMs(System.currentTimeMillis() - startTime);
        response.setTimestamp(LocalDateTime.now());

        // Cache the result
        cacheService.cacheResult(url, response);

        // Save to history
        DetectionHistory history = DetectionHistory.builder()
                .url(url)
                .isPhishing(response.getIsPhishing())
                .confidence(response.getConfidence() != null ? response.getConfidence() : 0.0)
                .riskLevel(convertRiskLevel(response.getRiskLevel().toString()))
                .features(response.getFeatures().toString())
                .detectedAt(response.getTimestamp())
                .processingTimeMs(response.getProcessingTimeMs())
                .build();
        historyService.save(history);

        log.info("Detection complete for URL: {} - isPhishing: {}, confidence: {}",
                url, response.getIsPhishing(), response.getConfidence() != null ? response.getConfidence() : "N/A");

        return response;
    }

    /**
     * Build detection response from ML service response
     */
    private DetectionResponse buildDetectionResponse(String url, MlServiceResponse mlResponse,
                                                     Map<String, Object> features) {
        // Handle null confidence - use 0.5 as default
        Double confidence = mlResponse.getConfidence();
        if (confidence == null) {
            confidence = mlResponse.getIsPhishing() ? 1.0 : 0.0;
            log.warn("ML service returned null confidence, using default: {}", confidence);
        }

        return DetectionResponse.builder()
                .url(url)
                .isPhishing(mlResponse.getIsPhishing())
                .confidence(confidence)
                .riskLevel(DetectionResponse.RiskLevel.fromConfidence(
                        mlResponse.getIsPhishing(),
                        confidence))
                .features(features)
                .fromCache(false)
                .timestamp(LocalDateTime.now())
                .build();
    }

    /**
     * Build detection response from heuristic analysis (ML service fallback)
     */
    private DetectionResponse buildHeuristicResponse(String url, double heuristicScore,
                                                     Map<String, Object> features) {
        boolean isPhishing = heuristicScore >= detectionThreshold;

        return DetectionResponse.builder()
                .url(url)
                .isPhishing(isPhishing)
                .confidence(heuristicScore)
                .riskLevel(DetectionResponse.RiskLevel.fromConfidence(isPhishing, heuristicScore))
                .features(features)
                .fromCache(false)
                .timestamp(LocalDateTime.now())
                .build();
    }

    /**
     * Batch detection for multiple URLs
     *
     * @param urls List of URLs to detect
     * @return Map of URL to detection response
     */
    public Map<String, DetectionResponse> detectBatch(java.util.List<String> urls) {
        log.info("Starting batch detection for {} URLs", urls.size());

        Map<String, DetectionResponse> results = new java.util.HashMap<>();
        for (String url : urls) {
            try {
                DetectionRequest request = DetectionRequest.builder()
                        .url(url)
                        .fetchContent(false)
                        .build();
                results.put(url, detectUrl(request));
            } catch (Exception e) {
                log.error("Failed to detect URL: {}", url, e);
            }
        }

        return results;
    }

    /**
     * Check if the ML service is healthy
     *
     * @return true if ML service is available
     */
    public boolean isMlServiceHealthy() {
        return mlServiceClient.healthCheck();
    }

    /**
     * Convert DetectionResponse RiskLevel string to DetectionHistory RiskLevel enum
     */
    private DetectionHistory.RiskLevel convertRiskLevel(String riskLevel) {
        if (riskLevel == null) return DetectionHistory.RiskLevel.LOW;

        switch (riskLevel.toUpperCase()) {
            case "CRITICAL":
            case "HIGH":
                return DetectionHistory.RiskLevel.HIGH;
            case "MEDIUM":
                return DetectionHistory.RiskLevel.MEDIUM;
            case "LOW":
                return DetectionHistory.RiskLevel.LOW;
            case "SAFE":
                return DetectionHistory.RiskLevel.SAFE;
            default:
                return DetectionHistory.RiskLevel.LOW;
        }
    }

    /**
     * Build response for whitelisted URLs (always safe)
     */
    private DetectionResponse buildWhitelistResponse(String url) {
        return DetectionResponse.builder()
                .url(url)
                .isPhishing(false)
                .confidence(1.0)
                .riskLevel(DetectionResponse.RiskLevel.SAFE)
                .fromCache(false)
                .timestamp(LocalDateTime.now())
                .processingTimeMs(0L)
                .build();
    }

    /**
     * Build response for blacklisted URLs (always phishing)
     */
    private DetectionResponse buildBlacklistResponse(String url) {
        return DetectionResponse.builder()
                .url(url)
                .isPhishing(true)
                .confidence(1.0)
                .riskLevel(DetectionResponse.RiskLevel.CRITICAL)
                .fromCache(false)
                .timestamp(LocalDateTime.now())
                .processingTimeMs(0L)
                .build();
    }
}

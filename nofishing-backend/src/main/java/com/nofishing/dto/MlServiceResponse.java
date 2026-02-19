package com.nofishing.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

/**
 * Response DTO from ML Service API
 *
 * @author NoFishing Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MlServiceResponse {

    /**
     * Whether the URL is classified as phishing
     */
    @JsonProperty("is_phishing")
    private Boolean isPhishing;

    /**
     * Confidence score (0.0 - 1.0)
     * Mapped from "probability" in ML API response
     */
    @JsonProperty("probability")
    private Double confidence;

    /**
     * Risk level classification
     */
    @JsonProperty("risk_level")
    private String riskLevel;

    /**
     * Extracted features from the URL
     */
    private Map<String, Object> features;

    /**
     * Processing time in milliseconds
     */
    @JsonProperty("processing_time_ms")
    private Long processingTimeMs;

    /**
     * Error message if classification failed
     */
    private String error;
}

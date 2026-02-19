package com.nofishing.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * Response DTO for URL detection endpoint
 *
 * @author NoFishing Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DetectionResponse {

    /**
     * The URL that was analyzed
     */
    private String url;

    /**
     * Whether the URL is classified as phishing
     */
    private Boolean isPhishing;

    /**
     * Confidence score (0.0 - 1.0)
     * Higher values indicate higher confidence in the classification
     */
    private Double confidence;

    /**
     * Risk level classification
     */
    private RiskLevel riskLevel;

    /**
     * Processing time in milliseconds
     */
    private Long processingTimeMs;

    /**
     * Timestamp of the detection
     */
    private LocalDateTime timestamp;

    /**
     * Additional features extracted from the URL
     */
    private Map<String, Object> features;

    /**
     * Whether the result was retrieved from cache
     */
    @Builder.Default
    private Boolean fromCache = false;

    /**
     * Risk level enumeration
     */
    public enum RiskLevel {
        SAFE("Safe", 0.0, 0.0),
        LOW("Low Risk", 0.0, 0.3),
        MEDIUM("Medium Risk", 0.3, 0.6),
        HIGH("High Risk", 0.6, 0.8),
        CRITICAL("Critical Risk", 0.8, 1.0);

        private final String displayName;
        private final double minThreshold;
        private final double maxThreshold;

        RiskLevel(String displayName, double minThreshold, double maxThreshold) {
            this.displayName = displayName;
            this.minThreshold = minThreshold;
            this.maxThreshold = maxThreshold;
        }

        public String getDisplayName() {
            return displayName;
        }

        public static RiskLevel fromConfidence(double confidence) {
            for (RiskLevel level : values()) {
                if (confidence >= level.minThreshold && confidence < level.maxThreshold) {
                    return level;
                }
            }
            return CRITICAL;
        }

        public static RiskLevel fromConfidence(boolean isPhishing, double confidence) {
            if (!isPhishing && confidence >= 0.95) {
                return SAFE;
            }
            return fromConfidence(confidence);
        }
    }
}

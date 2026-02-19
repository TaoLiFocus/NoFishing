package com.nofishing.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Request DTO for ML Service API
 *
 * @author NoFishing Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MlServiceRequest {

    /**
     * The URL to classify
     */
    private String url;

    /**
     * Whether to fetch and analyze page content
     */
    @Builder.Default
    private Boolean fetchContent = false;

    /**
     * User agent for content fetching
     */
    private String userAgent;

    /**
     * Timeout in milliseconds
     */
    private Integer timeout;
}

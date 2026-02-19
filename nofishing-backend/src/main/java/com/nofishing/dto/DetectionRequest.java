package com.nofishing.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Request DTO for URL detection endpoint
 *
 * @author NoFishing Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DetectionRequest {

    /**
     * The URL to be analyzed for phishing detection
     */
    @NotBlank(message = "URL cannot be blank")
    @Pattern(regexp = "^(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]",
            message = "Invalid URL format")
    private String url;

    /**
     * Whether to fetch and analyze the page content
     * Default: false for faster response
     */
    @Builder.Default
    private Boolean fetchContent = false;

    /**
     * Optional user agent for content fetching
     */
    private String userAgent;

    /**
     * Optional timeout for content fetching (milliseconds)
     */
    private Integer timeout;
}

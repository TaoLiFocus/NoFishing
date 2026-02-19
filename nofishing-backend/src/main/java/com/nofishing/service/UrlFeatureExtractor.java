package com.nofishing.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * URL Feature Extractor
 *
 * Extracts lexical and structural features from URLs for analysis
 *
 * @author NoFishing Team
 */
@Slf4j
@Component
public class UrlFeatureExtractor {

    // Patterns for suspicious URL detection
    private static final Pattern IP_ADDRESS_PATTERN = Pattern.compile(
            "^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$"
    );

    private static final Pattern SUSPICIOUS_WORDS_PATTERN = Pattern.compile(
            "(?i).*(login|signin|account|verify|secure|update|confirm|bank|wallet|crypto).*"
    );

    private static final Pattern BRAND_IMPERSONATION_PATTERN = Pattern.compile(
            "(?i).*(apple|google|microsoft|amazon|facebook|paypal|dropbox|netflix).*"
    );

    /**
     * Extract features from URL
     *
     * @param urlString The URL to analyze
     * @return Map of extracted features
     */
    public Map<String, Object> extractFeatures(String urlString) {
        Map<String, Object> features = new HashMap<>();

        try {
            URI uri = new URI(urlString);

            // Basic URL components
            features.put("url", urlString);
            features.put("scheme", uri.getScheme());
            features.put("host", uri.getHost());
            features.put("path", uri.getPath());
            features.put("length", urlString.length());

            // Domain features
            if (uri.getHost() != null) {
                String host = uri.getHost().toLowerCase();
                features.put("domainLength", host.length());
                features.put("subdomainCount", countSubdomains(host));
                features.put("hasIpAddress", hasIpAddress(host));
                features.put("hasHttps", "https".equals(uri.getScheme()));
            }

            // Path features
            if (uri.getPath() != null) {
                features.put("pathLength", uri.getPath().length());
                features.put("pathDepth", countPathDepth(uri.getPath()));
            }

            // Suspicious patterns
            features.put("hasSuspiciousWords", hasSuspiciousWords(urlString));
            features.put("hasBrandImpersonation", hasBrandImpersonation(urlString));
            features.put("hasAtSymbol", urlString.contains("@"));
            features.put("hasDashInDomain", hasDashInDomain(uri.getHost()));
            features.put("hasExcessiveDots", countDots(urlString) > 5);
            features.put("hasIpAddressPattern", IP_ADDRESS_PATTERN.matcher(uri.getHost() != null ? uri.getHost() : "").matches());

            // Query parameters
            features.put("hasQueryParams", uri.getQuery() != null);
            if (uri.getQuery() != null) {
                features.put("queryParamsCount", uri.getQuery().split("&").length);
            }

            // Port features
            features.put("hasNonStandardPort", uri.getPort() > 0 && uri.getPort() != 80 && uri.getPort() != 443);

            // Fragment
            features.put("hasFragment", uri.getFragment() != null);

            // Calculate heuristic score
            double heuristicScore = calculateHeuristicScore(features);
            features.put("heuristicScore", heuristicScore);

            log.debug("Extracted features for URL: {} - Features: {}", urlString, features);

        } catch (URISyntaxException e) {
            log.warn("Failed to parse URL: {}", urlString, e);
            features.put("error", "Invalid URL");
        }

        return features;
    }

    /**
     * Count the number of subdomains in the host
     */
    private int countSubdomains(String host) {
        if (host == null) return 0;
        String[] parts = host.split("\\.");
        return Math.max(0, parts.length - 2); // Subtract for TLD and domain
    }

    /**
     * Check if host contains an IP address
     */
    private boolean hasIpAddress(String host) {
        if (host == null) return false;
        return IP_ADDRESS_PATTERN.matcher(host).matches();
    }

    /**
     * Check if URL contains suspicious words
     */
    private boolean hasSuspiciousWords(String url) {
        return SUSPICIOUS_WORDS_PATTERN.matcher(url).matches();
    }

    /**
     * Check if URL contains brand impersonation
     */
    private boolean hasBrandImpersonation(String url) {
        return BRAND_IMPERSONATION_PATTERN.matcher(url).matches();
    }

    /**
     * Check if domain has dashes
     */
    private boolean hasDashInDomain(String host) {
        if (host == null) return false;
        return host.contains("-");
    }

    /**
     * Count dots in URL
     */
    private int countDots(String url) {
        return url == null ? 0 : url.length() - url.replace(".", "").length();
    }

    /**
     * Count path depth (number of / separators)
     */
    private int countPathDepth(String path) {
        if (path == null || path.isEmpty()) return 0;
        return path.split("/").length - 1;
    }

    /**
     * Calculate a heuristic phishing score based on extracted features
     * This is a simple rule-based score, NOT a machine learning prediction
     *
     * @return Score between 0 (legitimate) and 1 (phishing)
     */
    private double calculateHeuristicScore(Map<String, Object> features) {
        double score = 0.0;
        int totalChecks = 0;

        // Check for IP address (strong indicator)
        totalChecks++;
        if (Boolean.TRUE.equals(features.get("hasIpAddress"))) {
            score += 0.8;
        }

        // Check for HTTPS
        totalChecks++;
        if (Boolean.FALSE.equals(features.get("hasHttps"))) {
            score += 0.3;
        }

        // Check for suspicious words
        totalChecks++;
        if (Boolean.TRUE.equals(features.get("hasSuspiciousWords"))) {
            score += 0.4;
        }

        // Check for brand impersonation
        totalChecks++;
        if (Boolean.TRUE.equals(features.get("hasBrandImpersonation"))) {
            score += 0.6;
        }

        // Check for @ symbol
        totalChecks++;
        if (Boolean.TRUE.equals(features.get("hasAtSymbol"))) {
            score += 0.7;
        }

        // Check for excessive subdomains
        totalChecks++;
        Integer subdomainCount = (Integer) features.get("subdomainCount");
        if (subdomainCount != null && subdomainCount > 3) {
            score += 0.4;
        }

        // Check for non-standard port
        totalChecks++;
        if (Boolean.TRUE.equals(features.get("hasNonStandardPort"))) {
            score += 0.5;
        }

        // Check for dash in domain
        totalChecks++;
        if (Boolean.TRUE.equals(features.get("hasDashInDomain"))) {
            score += 0.2;
        }

        // Normalize score
        return score / totalChecks;
    }
}

package com.nofishing.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for UrlFeatureExtractor
 */
class UrlFeatureExtractorTest {

    private UrlFeatureExtractor extractor;

    @BeforeEach
    void setUp() {
        extractor = new UrlFeatureExtractor();
    }

    @Test
    void testExtractFeatures_NormalUrl() {
        String url = "https://www.example.com/page";
        Map<String, Object> features = extractor.extractFeatures(url);

        assertEquals(url, features.get("url"));
        assertEquals("https", features.get("scheme"));
        assertEquals("www.example.com", features.get("host"));
        assertEquals("/page", features.get("path"));
        assertEquals(Boolean.TRUE, features.get("hasHttps"));
    }

    @Test
    void testExtractFeatures_IpAddress() {
        String url = "http://192.168.1.1/login";
        Map<String, Object> features = extractor.extractFeatures(url);

        assertEquals(Boolean.TRUE, features.get("hasIpAddress"));
    }

    @Test
    void testExtractFeatures_SuspiciousWords() {
        String url = "http://example.com/verify-account";
        Map<String, Object> features = extractor.extractFeatures(url);

        assertEquals(Boolean.TRUE, features.get("hasSuspiciousWords"));
    }

    @Test
    void testCalculateHeuristicScore_HighRisk() {
        String url = "http://192.168.1.1@fake-paypal.com/login";
        Map<String, Object> features = extractor.extractFeatures(url);

        Double score = (Double) features.get("heuristicScore");
        assertNotNull(score);
        assertTrue(score > 0.5, "High-risk URL should have heuristic score > 0.5");
    }
}

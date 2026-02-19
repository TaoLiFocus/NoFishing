package com.nofishing.service;

import com.nofishing.dto.DetectionResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.CachePut;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * URL Cache Service
 *
 * Handles caching of detection results using Redis
 *
 * @author NoFishing Team
 */
@Slf4j
@Service
public class UrlCacheService {

    private static final String CACHE_NAME = "detectionCache";

    /**
     * Get cached detection result for a URL
     *
     * @param url The URL to check
     * @return Cached detection response, or null if not found
     */
    @Cacheable(value = CACHE_NAME, key = "#url", unless = "#result == null")
    public DetectionResponse getCachedResult(String url) {
        log.debug("Cache MISS for URL: {}", url);
        return null;
    }

    /**
     * Cache a detection result
     *
     * @param url The URL
     * @param response The detection response to cache
     * @return The cached response
     */
    @CachePut(value = CACHE_NAME, key = "#url")
    public DetectionResponse cacheResult(String url, DetectionResponse response) {
        log.debug("Cached result for URL: {}", url);
        return response;
    }

    /**
     * Evict a URL from cache
     *
     * @param url The URL to evict
     */
    @CacheEvict(value = CACHE_NAME, key = "#url")
    public void evictFromCache(String url) {
        log.debug("Evicted from cache: {}", url);
    }

    /**
     * Evict all entries from cache
     */
    @CacheEvict(value = CACHE_NAME, allEntries = true)
    public void evictAllFromCache() {
        log.debug("Evicted all entries from cache");
    }

    /**
     * Generate cache key for a URL
     * Uses MD5 hash for consistent key generation
     *
     * @param url The URL
     * @return Cache key
     */
    public String generateCacheKey(String url) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(url.toLowerCase().getBytes());
            StringBuilder sb = new StringBuilder();
            for (byte b : digest) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            log.error("Failed to generate cache key", e);
            return String.valueOf(url.hashCode());
        }
    }
}

package com.nofishing.controller;

import com.nofishing.dto.DetectionRequest;
import com.nofishing.dto.DetectionResponse;
import com.nofishing.service.DetectionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Detection API Controller
 *
 * REST API endpoints for phishing detection
 *
 * @author NoFishing Team
 */
@Slf4j
@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
public class DetectionController {

    private final DetectionService detectionService;

    /**
     * Detect if a URL is phishing
     *
     * POST /api/v1/detect
     *
     * @param request Detection request with URL and options
     * @return Detection response with classification result
     */
    @PostMapping("/detect")
    public ResponseEntity<DetectionResponse> detect(@Valid @RequestBody DetectionRequest request) {
        log.info("Received detection request for URL: {}", request.getUrl());

        DetectionResponse response = detectionService.detectUrl(request);

        return ResponseEntity.ok(response);
    }

    /**
     * Batch detection for multiple URLs
     *
     * POST /api/v1/detect/batch
     *
     * @param requestBody Map with 'urls' key containing list of URLs
     * @return Map of URL to detection response
     */
    @PostMapping("/detect/batch")
    public ResponseEntity<Map<String, DetectionResponse>> detectBatch(
            @RequestBody Map<String, List<String>> requestBody) {

        List<String> urls = requestBody.get("urls");
        if (urls == null || urls.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        log.info("Received batch detection request for {} URLs", urls.size());

        Map<String, DetectionResponse> responses = detectionService.detectBatch(urls);

        return ResponseEntity.ok(responses);
    }

    /**
     * Quick check endpoint (simplified response)
     *
     * GET /api/v1/check?url={url}
     *
     * @param url The URL to check
     * @return Simplified boolean response
     */
    @GetMapping("/check")
    public ResponseEntity<Map<String, Object>> check(@RequestParam String url) {
        log.info("Received quick check request for URL: {}", url);

        DetectionRequest request = DetectionRequest.builder()
                .url(url)
                .fetchContent(false)
                .build();

        DetectionResponse response = detectionService.detectUrl(request);

        Map<String, Object> result = Map.of(
                "url", url,
                "isPhishing", response.getIsPhishing(),
                "confidence", response.getConfidence(),
                "riskLevel", response.getRiskLevel().toString()
        );

        return ResponseEntity.ok(result);
    }
}

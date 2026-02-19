package com.nofishing.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nofishing.config.MlServiceConfig;
import com.nofishing.dto.MlServiceRequest;
import com.nofishing.dto.MlServiceResponse;
import com.nofishing.exception.MlServiceException;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * HTTP Client for ML Service Communication
 *
 * Handles communication with the Flask + PyTorch ML API service
 * for phishing classification.
 *
 * @author NoFishing Team
 */
@Slf4j
@Component
public class MlServiceClient {

    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final MlServiceConfig mlServiceConfig;
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    public MlServiceClient(OkHttpClient httpClient, ObjectMapper objectMapper, MlServiceConfig mlServiceConfig) {
        this.httpClient = httpClient;
        this.objectMapper = objectMapper;
        this.mlServiceConfig = mlServiceConfig;
    }

    /**
     * Classify a URL using the ML service
     *
     * @param url The URL to classify
     * @param fetchContent Whether to fetch and analyze page content
     * @return ML Service response with classification results
     * @throws MlServiceException if the ML service call fails
     */
    public MlServiceResponse classifyUrl(String url, boolean fetchContent) {
        long startTime = System.currentTimeMillis();

        MlServiceRequest request = MlServiceRequest.builder()
                .url(url)
                .fetchContent(fetchContent)
                .build();

        try {
            String jsonBody = objectMapper.writeValueAsString(request);
            Request httpRequest = new Request.Builder()
                    .url(mlServiceConfig.getMlServiceBaseUrl() + "/classify")
                    .post(RequestBody.create(jsonBody, JSON))
                    .addHeader("Content-Type", "application/json")
                    .addHeader("Accept", "application/json")
                    .build();

            log.debug("Sending ML service request for URL: {}", url);

            try (Response response = httpClient.newCall(httpRequest).execute()) {
                long processingTime = System.currentTimeMillis() - startTime;

                if (!response.isSuccessful()) {
                    String errorBody = response.body() != null ? response.body().string() : "Unknown error";
                    log.error("ML service returned error: {} - {}", response.code(), errorBody);
                    throw new MlServiceException("ML service returned error: " + response.code() + " - " + errorBody);
                }

                String responseBody = response.body() != null ? response.body().string() : "{}";
                MlServiceResponse mlResponse = objectMapper.readValue(responseBody, MlServiceResponse.class);
                mlResponse.setProcessingTimeMs(processingTime);

                log.debug("ML service response received in {}ms - isPhishing: {}, confidence: {}",
                        processingTime, mlResponse.getIsPhishing(), mlResponse.getConfidence());

                return mlResponse;
            }

        } catch (IOException e) {
            log.error("Failed to communicate with ML service for URL: {}", url, e);
            throw new MlServiceException("Failed to communicate with ML service: " + e.getMessage(), e);
        }
    }

    /**
     * Health check for the ML service
     *
     * @return true if the ML service is healthy, false otherwise
     */
    public boolean healthCheck() {
        Request request = new Request.Builder()
                .url(mlServiceConfig.getMlServiceBaseUrl() + "/health")
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return response.isSuccessful();
        } catch (IOException e) {
            log.warn("ML service health check failed: {}", e.getMessage());
            return false;
        }
    }
}

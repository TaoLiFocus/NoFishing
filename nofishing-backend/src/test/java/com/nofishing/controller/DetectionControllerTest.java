package com.nofishing.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nofishing.dto.DetectionRequest;
import com.nofishing.dto.DetectionResponse;
import com.nofishing.service.DetectionService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Unit tests for DetectionController
 */
@WebMvcTest(DetectionController.class)
class DetectionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private DetectionService detectionService;

    @Test
    void testDetect_Success() throws Exception {
        DetectionRequest request = DetectionRequest.builder()
                .url("http://test.com")
                .fetchContent(false)
                .build();

        DetectionResponse response = DetectionResponse.builder()
                .url("http://test.com")
                .isPhishing(false)
                .confidence(0.1)
                .timestamp(LocalDateTime.now())
                .build();

        when(detectionService.detectUrl(any(DetectionRequest.class)))
                .thenReturn(response);

        mockMvc.perform(post("/api/v1/detect")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.url").value("http://test.com"))
                .andExpect(jsonPath("$.isPhishing").value(false));
    }

    @Test
    void testDetect_InvalidUrl() throws Exception {
        DetectionRequest request = DetectionRequest.builder()
                .url("not-a-valid-url")
                .build();

        mockMvc.perform(post("/api/v1/detect")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest());
    }

    @Test
    void testCheck_Success() throws Exception {
        DetectionResponse response = DetectionResponse.builder()
                .url("http://test.com")
                .isPhishing(false)
                .confidence(0.1)
                .timestamp(LocalDateTime.now())
                .build();

        when(detectionService.detectUrl(any(DetectionRequest.class)))
                .thenReturn(response);

        mockMvc.perform(get("/api/v1/check")
                        .param("url", "http://test.com"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.isPhishing").value(false));
    }
}

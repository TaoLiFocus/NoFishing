package com.nofishing.config;

import okhttp3.OkHttpClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

/**
 * ML Service Client Configuration
 *
 * @author NoFishing Team
 */
@Configuration
public class MlServiceConfig {

    @Value("${ml-service.base-url}")
    private String mlServiceBaseUrl;

    @Value("${ml-service.connect-timeout:1000}")
    private int connectTimeout;

    @Value("${ml-service.read-timeout:2000}")
    private int readTimeout;

    @Bean
    public OkHttpClient okHttpClient() {
        return new OkHttpClient.Builder()
                .connectTimeout(Duration.ofMillis(connectTimeout))
                .readTimeout(Duration.ofMillis(readTimeout))
                .writeTimeout(Duration.ofMillis(readTimeout))
                .retryOnConnectionFailure(true)
                .build();
    }

    public String getMlServiceBaseUrl() {
        return mlServiceBaseUrl;
    }
}

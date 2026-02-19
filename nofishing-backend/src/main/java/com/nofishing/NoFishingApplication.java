package com.nofishing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.EnableAspectJAutoProxy;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;
import org.springframework.scheduling.annotation.EnableAsync;

/**
 * NoFishing Phishing Detection System - Main Application Entry Point
 *
 * This is the main Spring Boot application class that bootstraps the
 * phishing website detection backend service.
 */
@SpringBootApplication
@EnableAspectJAutoProxy(proxyTargetClass = true)
@EnableJpaAuditing
@EnableAsync
public class NoFishingApplication {

    public static void main(String[] args) {
        SpringApplication.run(NoFishingApplication.class, args);
    }
}


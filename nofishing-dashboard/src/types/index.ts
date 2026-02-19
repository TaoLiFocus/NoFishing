// API Types
export interface User {
  id: number;
  username: string;
  email?: string;
  phone?: string;
  realName?: string;
  role: 'ADMIN' | 'USER';
  enabled: boolean;
  createdAt: string;
  updatedAt: string;
  lastLoginAt?: string;
  lastLoginIp?: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  accessToken: string;
  refreshToken: string;
  userId: number;
  username: string;
  role: string;
}

export interface DetectionRequest {
  url: string;
}

export interface DetectionResponse {
  url: string;
  isPhishing: boolean;
  confidence: number;
  features?: Record<string, number>;
  riskLevel: 'HIGH' | 'MEDIUM' | 'LOW' | 'SAFE';
  timestamp: string;
}

export interface DetectionHistory {
  id: number;
  url: string;
  isPhishing: boolean;
  confidence?: number;
  features?: string;
  source?: string;
  ipAddress?: string;
  userAgent?: string;
  detectedAt: string;
  riskLevel?: 'HIGH' | 'MEDIUM' | 'LOW' | 'SAFE';
}

export interface WhitelistEntry {
  id: number;
  pattern: string;
  type?: string;
  enabled: boolean;
  comment?: string;
  addedBy?: string;
  createdAt: string;
  expiresAt?: string;
}

export interface BlacklistEntry {
  id: number;
  pattern: string;
  type?: string;
  enabled: boolean;
  comment?: string;
  addedBy?: string;
  threatType?: string;
  createdAt: string;
  expiresAt?: string;
}

export interface PaginatedResponse<T> {
  content: T[];
  pageNumber: number;
  pageSize: number;
  totalElements: number;
  totalPages: number;
  first: boolean;
  last: boolean;
}

export interface StatisticsSummary {
  totalDetections: number;
  phishingCount: number;
  safeCount: number;
  phishingRate: number;
}

export interface TrendData {
  date: string;
  total: number;
  phishing: number;
  safe: number;
}

export interface RiskDistribution {
  HIGH: number;
  MEDIUM: number;
  LOW: number;
  SAFE: number;
}

export interface SystemStatus {
  status: 'healthy' | 'degraded' | 'down';
  uptime: number;
  memory: {
    used: number;
    total: number;
    percentage: number;
  };
  cpu: number;
  services: {
    ml: boolean;
    redis: boolean;
    database: boolean;
  };
}

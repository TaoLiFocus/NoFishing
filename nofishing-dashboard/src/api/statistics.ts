import apiClient from './client';
import type {
  StatisticsSummary,
  TrendData,
  RiskDistribution,
} from '../types';

export const statisticsApi = {
  getSummary: () =>
    apiClient.get<StatisticsSummary>('/api/v1/statistics/summary'),

  getTrend: (params?: { startTime?: string; endTime?: string }) =>
    apiClient.get<TrendData[]>('/api/v1/statistics/trend', { params }),

  getDistribution: () =>
    apiClient.get<RiskDistribution>('/api/v1/statistics/distribution'),
};

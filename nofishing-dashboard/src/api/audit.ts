import { apiClient } from './client';

export interface AuditLog {
  id: number;
  operation: string;
  module: string;
  operatedBy: string;
  targetType?: string;
  targetId?: number;
  targetValue?: string;
  ipAddress?: string;
  userAgent?: string;
  status?: string;
  errorMessage?: string;
  createdAt: string;
}

export interface PaginatedAuditLogs {
  content: AuditLog[];
  pageNumber: number;
  pageSize: number;
  totalElements: number;
  totalPages: number;
  first: boolean;
  last: boolean;
}

export const auditApi = {
  // Search audit logs
  searchLogs: (params: {
    operation?: string;
    module?: string;
    operatedBy?: string;
    startDate?: string;
    endDate?: string;
    page?: number;
    size?: number;
  }) => apiClient.get<PaginatedAuditLogs>('/api/v1/admin/audit-logs', { params }),

  // Get recent logs
  getRecentLogs: (limit = 50) =>
    apiClient.get<AuditLog[]>('/api/v1/admin/audit-logs/recent', { params: { limit } }),

  // Get logs by operation
  getByOperation: (operation: string, page = 0, size = 20) =>
    apiClient.get<PaginatedAuditLogs>(`/api/v1/admin/audit-logs/operation/${operation}`, {
      params: { page, size }
    }),

  // Get logs by module
  getByModule: (module: string, page = 0, size = 20) =>
    apiClient.get<PaginatedAuditLogs>(`/api/v1/admin/audit-logs/module/${module}`, {
      params: { page, size }
    }),

  // Get logs by operator
  getByOperatedBy: (operatedBy: string, page = 0, size = 20) =>
    apiClient.get<PaginatedAuditLogs>(`/api/v1/admin/audit-logs/operator/${operatedBy}`, {
      params: { page, size }
    }),

  // Get logs by date range
  getByDateRange: (startDate: string, endDate: string, page = 0, size = 20) =>
    apiClient.get<PaginatedAuditLogs>('/api/v1/admin/audit-logs/date-range', {
      params: { startDate, endDate, page, size }
    }),
};

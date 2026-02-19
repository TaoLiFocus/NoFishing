import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Spin } from 'antd';
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { statisticsApi } from '../api/statistics';
import type { TrendData, RiskDistribution } from '../types';
import { useRequireAuth } from '../hooks/useRequireAuth';
import dayjs from 'dayjs';

const COLORS = {
  HIGH: '#cf1322',
  MEDIUM: '#fa8c16',
  LOW: '#faad14',
  SAFE: '#52c41a',
};

const Statistics: React.FC = () => {
  useRequireAuth();
  const [trendData, setTrendData] = useState<TrendData[]>([]);
  const [riskDistribution, setRiskDistribution] = useState<RiskDistribution | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [trendRes, distRes] = await Promise.all([
          statisticsApi.getTrend(),
          statisticsApi.getDistribution(),
        ]);
        setTrendData(trendRes.data);
        setRiskDistribution(distRes.data);
      } catch (error) {
        console.error('Failed to fetch statistics:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 0' }}>
        <Spin size="large" />
      </div>
    );
  }

  const pieData = riskDistribution
    ? Object.entries(riskDistribution)
        .filter(([_, value]) => value > 0)
        .map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ marginBottom: 32, fontSize: 28, fontWeight: 600 }}>统计分析</h1>

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={16}>
          <Card title="检测趋势（近7天）">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(date) => dayjs(date).format('MM/DD')}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="total"
                  stroke="#8884d8"
                  strokeWidth={2}
                  name="总计"
                />
                <Line
                  type="monotone"
                  dataKey="phishing"
                  stroke="#cf1322"
                  strokeWidth={2}
                  name="钓鱼"
                />
                <Line
                  type="monotone"
                  dataKey="safe"
                  stroke="#52c41a"
                  strokeWidth={2}
                  name="安全"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="风险分布" style={{ height: '100%' }}>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  nameKey="name"
                  dataKey="value"
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  label={false}
                >
                  {pieData.map((entry, index) => (
                    <Cell
                      key={`cell-${entry.name}-${index}`}
                      fill={COLORS[entry.name as keyof typeof COLORS] || '#888'}
                    />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ marginTop: 16 }}>
              {pieData.map((entry) => (
                <div key={entry.name} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ color: COLORS[entry.name as keyof typeof COLORS] }}>
                    {entry.name}
                  </span>
                  <strong>{entry.value}</strong>
                </div>
              ))}
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Statistics;

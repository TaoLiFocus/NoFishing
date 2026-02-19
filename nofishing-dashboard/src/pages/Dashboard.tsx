import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Spin } from 'antd';
import {
  SafetyOutlined,
  CheckCircleOutlined as ShieldCheckOutlined,
  SecurityScanOutlined as ShieldOutlined,
  AlertOutlined,
} from '@ant-design/icons';
import { statisticsApi } from '../api/statistics';
import type { StatisticsSummary } from '../types';
import { useRequireAuth } from '../hooks/useRequireAuth';

const Dashboard: React.FC = () => {
  useRequireAuth();
  const [summary, setSummary] = useState<StatisticsSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await statisticsApi.getSummary();
        setSummary(response.data);
      } catch (error) {
        console.error('Failed to fetch summary:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchSummary();
  }, []);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 0' }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ marginBottom: 32, fontSize: 28, fontWeight: 600 }}>仪表盘</h1>
      <Row gutter={[24, 24]}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="总检测数"
              value={summary?.totalDetections || 0}
              prefix={<SafetyOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="钓鱼网站"
              value={summary?.phishingCount || 0}
              prefix={<AlertOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="安全网站"
              value={summary?.safeCount || 0}
              prefix={<ShieldCheckOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="钓鱼率"
              value={summary?.phishingRate || 0}
              suffix="%"
              prefix={<ShieldOutlined />}
              precision={2}
              valueStyle={{
                color: (summary?.phishingRate || 0) > 50 ? '#cf1322' : '#52c41a'
              }}
            />
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 32 }} title="快捷操作">
        <p style={{ color: '#888', marginBottom: 16, fontSize: 14, lineHeight: 1.6 }}>
          请从菜单中选择一个功能开始使用：
        </p>
        <ul style={{ lineHeight: 2, fontSize: 14, paddingLeft: 20 }}>
          <li><strong>URL检测:</strong> 检查URL是否为钓鱼网站</li>
          <li><strong>历史记录:</strong> 查看检测历史</li>
          <li><strong>统计分析:</strong> 查看检测趋势和分析</li>
          <li><strong>白名单/黑名单:</strong> 管理信任和阻止的域名</li>
        </ul>
      </Card>
    </div>
  );
};

export default Dashboard;

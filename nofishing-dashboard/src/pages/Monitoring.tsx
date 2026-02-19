import React from 'react';
import { Card, Row, Col, Progress, Tag, Statistic } from 'antd';
import {
  CloudServerOutlined,
  DatabaseOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useRequireAuth } from '../hooks/useRequireAuth';

const Monitoring: React.FC = () => {
  useRequireAuth();

  const systemHealth = {
    status: 'healthy',
    uptime: 99.9,
    services: {
      ml: true,
      redis: true,
      database: true,
    },
    memory: {
      used: 512,
      total: 2048,
      percentage: 25,
    },
    cpu: 35,
  };

  const getStatusTag = (status: boolean) => (
    <Tag
      icon={status ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
      color={status ? 'success' : 'error'}
    >
      {status ? '运行中' : '离线'}
    </Tag>
  );

  return (
    <div>
      <h2>系统监控</h2>

      <Row gutter={16}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="系统状态"
              value="健康"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="运行时间"
              value={systemHealth.uptime}
              suffix="%"
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card title="CPU使用率">
            <Progress
              type="circle"
              percent={systemHealth.cpu}
              status={systemHealth.cpu > 80 ? 'exception' : 'active'}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card title="内存使用率">
            <Progress
              type="circle"
              percent={systemHealth.memory.percentage}
              format={() =>
                `${systemHealth.memory.used}MB / ${systemHealth.memory.total}MB`
              }
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="服务状态">
            <Row gutter={16}>
              <Col xs={24} md={8}>
                <div className="service-status">
                  <CloudServerOutlined style={{ fontSize: '32px', color: '#52c41a' }} />
                  <div className="service-info">
                    <div className="service-name">ML服务</div>
                    {getStatusTag(systemHealth.services.ml)}
                  </div>
                </div>
              </Col>
              <Col xs={24} md={8}>
                <div className="service-status">
                  <DatabaseOutlined style={{ fontSize: '32px', color: '#52c41a' }} />
                  <div className="service-info">
                    <div className="service-name">Redis缓存</div>
                    {getStatusTag(systemHealth.services.redis)}
                  </div>
                </div>
              </Col>
              <Col xs={24} md={8}>
                <div className="service-status">
                  <ApiOutlined style={{ fontSize: '32px', color: '#52c41a' }} />
                  <div className="service-info">
                    <div className="service-name">数据库 (H2)</div>
                    {getStatusTag(systemHealth.services.database)}
                  </div>
                </div>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 16 }} title="环境信息">
        <Row gutter={16}>
          <Col xs={24} md={12}>
            <div>
              <strong>应用:</strong> NoFishing Backend
            </div>
            <div>
              <strong>版本:</strong> 1.0.0
            </div>
            <div>
              <strong>Java:</strong> 17
            </div>
          </Col>
          <Col xs={24} md={12}>
            <div>
              <strong>Spring Boot:</strong> 3.2.2
            </div>
            <div>
              <strong>构建:</strong> {import.meta.env.VITE_BUILD_TIME || 'N/A'}
            </div>
            <div>
              <strong>环境:</strong> 开发环境
            </div>
          </Col>
        </Row>
      </Card>

      <style>{`
        .service-status {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 16px;
          background: #fafafa;
          border-radius: 8px;
        }
        .service-info {
          flex: 1;
        }
        .service-name {
          font-weight: 500;
          margin-bottom: 8px;
        }
      `}</style>
    </div>
  );
};

export default Monitoring;

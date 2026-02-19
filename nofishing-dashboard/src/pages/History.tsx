import React, { useEffect, useState } from 'react';
import { Table, Card, Input, Button, Space, Tag, Select } from 'antd';
import { SearchOutlined, ReloadOutlined, DownloadOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { historyApi } from '../api/detection';
import type { DetectionHistory } from '../types';
import { useRequireAuth } from '../hooks/useRequireAuth';
import { useAuth } from '../hooks/useAuth';

const History: React.FC = () => {
  useRequireAuth();
  const { isAdmin } = useAuth();
  const [data, setData] = useState<DetectionHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [pagination, setPagination] = useState({ current: 1, pageSize: 10, total: 0 });
  const [keyword, setKeyword] = useState('');
  const [isPhishing, setIsPhishing] = useState<boolean | undefined>();

  const fetchHistory = async (page = 1, size = 10) => {
    setLoading(true);
    try {
      const response = await historyApi.getHistory({
        page: page - 1,
        size,
        keyword: keyword || undefined,
        isPhishing,
      });
      setData(response.data.content);
      setPagination({
        current: page,
        pageSize: size,
        total: response.data.totalElements,
      });
    } catch (error) {
      console.error('Failed to fetch history:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSearch = () => {
    fetchHistory(1, pagination.pageSize);
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      await historyApi.exportHistory({
        keyword: keyword || undefined,
        isPhishing,
      });
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  const getRiskColor = (level?: string) => {
    switch (level) {
      case 'HIGH': return 'red';
      case 'MEDIUM': return 'orange';
      case 'LOW': return 'gold';
      case 'SAFE': return 'green';
      default: return 'default';
    }
  };

  const columns: ColumnsType<DetectionHistory> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: 'URL',
      dataIndex: 'url',
      key: 'url',
      ellipsis: true,
    },
    {
      title: '结果',
      dataIndex: 'isPhishing',
      key: 'isPhishing',
      width: 120,
      render: (isPhishing: boolean) => (
        <Tag color={isPhishing ? 'red' : 'green'}>
          {isPhishing ? '钓鱼' : '安全'}
        </Tag>
      ),
    },
    {
      title: '风险等级',
      dataIndex: 'riskLevel',
      key: 'riskLevel',
      width: 120,
      render: (level?: string) => (
        level ? <Tag color={getRiskColor(level)}>{level}</Tag> : '-'
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence?: number) =>
        confidence ? `${Math.round(confidence * 100)}%` : '-',
    },
    {
      title: '检测时间',
      dataIndex: 'detectedAt',
      key: 'detectedAt',
      width: 180,
      render: (date: string) => new Date(date).toLocaleString(),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ marginBottom: 32, fontSize: 28, fontWeight: 600 }}>检测历史</h1>

      <Card>
        <Space style={{ marginBottom: 24 }} wrap>
          <Input
            placeholder="按URL搜索"
            prefix={<SearchOutlined />}
            style={{ width: 300 }}
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onPressEnter={handleSearch}
            allowClear
          />
          <Select
            placeholder="按结果筛选"
            style={{ width: 150 }}
            value={isPhishing}
            onChange={setIsPhishing}
            allowClear
          >
            <Select.Option value={true}>钓鱼</Select.Option>
            <Select.Option value={false}>安全</Select.Option>
          </Select>
          <Button type="primary" onClick={handleSearch}>
            搜索
          </Button>
          <Button icon={<ReloadOutlined />} onClick={() => fetchHistory()}>
            刷新
          </Button>
          {isAdmin() && (
            <Button
              icon={<DownloadOutlined />}
              onClick={handleExport}
              loading={exporting}
            >
              导出Excel
            </Button>
          )}
        </Space>

        <Table
          columns={columns}
          dataSource={data}
          loading={loading}
          rowKey="id"
          pagination={{
            ...pagination,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条`,
            onChange: (page, pageSize) => fetchHistory(page, pageSize),
          }}
          scroll={{ x: 900 }}
        />
      </Card>
    </div>
  );
};

export default History;

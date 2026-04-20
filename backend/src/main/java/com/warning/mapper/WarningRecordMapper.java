package com.warning.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.warning.entity.WarningRecord;
import org.apache.ibatis.annotations.Mapper;

/**
 * 预警记录Mapper
 */
@Mapper
public interface WarningRecordMapper extends BaseMapper<WarningRecord> {
}

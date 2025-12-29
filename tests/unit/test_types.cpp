/**
 * Unit tests for nperf/types.h
 *
 * Tests all enum parsing, naming functions, and type size calculations.
 */

#include <gtest/gtest.h>
#include "nperf/types.h"
#include "test_utils.h"

namespace nperf {
namespace testing {

// ============================================================================
// CollectiveOp Tests
// ============================================================================

class CollectiveOpTest : public ::testing::Test {};

TEST_F(CollectiveOpTest, NameReturnsCorrectStrings) {
    EXPECT_STREQ(collectiveOpName(CollectiveOp::AllReduce), "allreduce");
    EXPECT_STREQ(collectiveOpName(CollectiveOp::AllGather), "allgather");
    EXPECT_STREQ(collectiveOpName(CollectiveOp::Broadcast), "broadcast");
    EXPECT_STREQ(collectiveOpName(CollectiveOp::Reduce), "reduce");
    EXPECT_STREQ(collectiveOpName(CollectiveOp::ReduceScatter), "reducescatter");
    EXPECT_STREQ(collectiveOpName(CollectiveOp::AlltoAll), "alltoall");
    EXPECT_STREQ(collectiveOpName(CollectiveOp::Gather), "gather");
    EXPECT_STREQ(collectiveOpName(CollectiveOp::Scatter), "scatter");
    EXPECT_STREQ(collectiveOpName(CollectiveOp::SendRecv), "sendrecv");
}

TEST_F(CollectiveOpTest, ParseAllReduce) {
    EXPECT_EQ(parseCollectiveOp("allreduce"), CollectiveOp::AllReduce);
    EXPECT_EQ(parseCollectiveOp("all-reduce"), CollectiveOp::AllReduce);
    EXPECT_EQ(parseCollectiveOp("all_reduce"), CollectiveOp::AllReduce);
}

TEST_F(CollectiveOpTest, ParseAllGather) {
    EXPECT_EQ(parseCollectiveOp("allgather"), CollectiveOp::AllGather);
    EXPECT_EQ(parseCollectiveOp("all-gather"), CollectiveOp::AllGather);
    EXPECT_EQ(parseCollectiveOp("all_gather"), CollectiveOp::AllGather);
}

TEST_F(CollectiveOpTest, ParseBroadcast) {
    EXPECT_EQ(parseCollectiveOp("broadcast"), CollectiveOp::Broadcast);
}

TEST_F(CollectiveOpTest, ParseReduce) {
    EXPECT_EQ(parseCollectiveOp("reduce"), CollectiveOp::Reduce);
}

TEST_F(CollectiveOpTest, ParseReduceScatter) {
    EXPECT_EQ(parseCollectiveOp("reducescatter"), CollectiveOp::ReduceScatter);
    EXPECT_EQ(parseCollectiveOp("reduce-scatter"), CollectiveOp::ReduceScatter);
    EXPECT_EQ(parseCollectiveOp("reduce_scatter"), CollectiveOp::ReduceScatter);
}

TEST_F(CollectiveOpTest, ParseAlltoAll) {
    EXPECT_EQ(parseCollectiveOp("alltoall"), CollectiveOp::AlltoAll);
    EXPECT_EQ(parseCollectiveOp("all-to-all"), CollectiveOp::AlltoAll);
    EXPECT_EQ(parseCollectiveOp("all_to_all"), CollectiveOp::AlltoAll);
}

TEST_F(CollectiveOpTest, ParseGather) {
    EXPECT_EQ(parseCollectiveOp("gather"), CollectiveOp::Gather);
}

TEST_F(CollectiveOpTest, ParseScatter) {
    EXPECT_EQ(parseCollectiveOp("scatter"), CollectiveOp::Scatter);
}

TEST_F(CollectiveOpTest, ParseSendRecv) {
    EXPECT_EQ(parseCollectiveOp("sendrecv"), CollectiveOp::SendRecv);
    EXPECT_EQ(parseCollectiveOp("send-recv"), CollectiveOp::SendRecv);
    EXPECT_EQ(parseCollectiveOp("send_recv"), CollectiveOp::SendRecv);
}

TEST_F(CollectiveOpTest, ParseInvalidReturnsDefault) {
    EXPECT_EQ(parseCollectiveOp(""), CollectiveOp::AllReduce);
    EXPECT_EQ(parseCollectiveOp("invalid"), CollectiveOp::AllReduce);
    EXPECT_EQ(parseCollectiveOp("ALLREDUCE"), CollectiveOp::AllReduce);  // Case sensitive
}

TEST_F(CollectiveOpTest, RoundTrip) {
    // Verify that parsing the name gives back the original op
    for (auto op : {CollectiveOp::AllReduce, CollectiveOp::AllGather,
                    CollectiveOp::Broadcast, CollectiveOp::Reduce,
                    CollectiveOp::ReduceScatter, CollectiveOp::AlltoAll,
                    CollectiveOp::Gather, CollectiveOp::Scatter,
                    CollectiveOp::SendRecv}) {
        EXPECT_EQ(parseCollectiveOp(collectiveOpName(op)), op);
    }
}

// ============================================================================
// DataType Tests
// ============================================================================

class DataTypeTest : public ::testing::Test {};

TEST_F(DataTypeTest, NameReturnsCorrectStrings) {
    EXPECT_STREQ(dataTypeName(DataType::Float32), "float32");
    EXPECT_STREQ(dataTypeName(DataType::Float64), "float64");
    EXPECT_STREQ(dataTypeName(DataType::Float16), "float16");
    EXPECT_STREQ(dataTypeName(DataType::BFloat16), "bfloat16");
    EXPECT_STREQ(dataTypeName(DataType::Int8), "int8");
    EXPECT_STREQ(dataTypeName(DataType::UInt8), "uint8");
    EXPECT_STREQ(dataTypeName(DataType::Int32), "int32");
    EXPECT_STREQ(dataTypeName(DataType::UInt32), "uint32");
    EXPECT_STREQ(dataTypeName(DataType::Int64), "int64");
    EXPECT_STREQ(dataTypeName(DataType::UInt64), "uint64");
}

TEST_F(DataTypeTest, SizeReturnsCorrectBytes) {
    EXPECT_EQ(dataTypeSize(DataType::Float32), 4u);
    EXPECT_EQ(dataTypeSize(DataType::Float64), 8u);
    EXPECT_EQ(dataTypeSize(DataType::Float16), 2u);
    EXPECT_EQ(dataTypeSize(DataType::BFloat16), 2u);
    EXPECT_EQ(dataTypeSize(DataType::Int8), 1u);
    EXPECT_EQ(dataTypeSize(DataType::UInt8), 1u);
    EXPECT_EQ(dataTypeSize(DataType::Int32), 4u);
    EXPECT_EQ(dataTypeSize(DataType::UInt32), 4u);
    EXPECT_EQ(dataTypeSize(DataType::Int64), 8u);
    EXPECT_EQ(dataTypeSize(DataType::UInt64), 8u);
}

TEST_F(DataTypeTest, ParseFloat32) {
    EXPECT_EQ(parseDataType("float32"), DataType::Float32);
    EXPECT_EQ(parseDataType("float"), DataType::Float32);
    EXPECT_EQ(parseDataType("f32"), DataType::Float32);
}

TEST_F(DataTypeTest, ParseFloat64) {
    EXPECT_EQ(parseDataType("float64"), DataType::Float64);
    EXPECT_EQ(parseDataType("double"), DataType::Float64);
    EXPECT_EQ(parseDataType("f64"), DataType::Float64);
}

TEST_F(DataTypeTest, ParseFloat16) {
    EXPECT_EQ(parseDataType("float16"), DataType::Float16);
    EXPECT_EQ(parseDataType("half"), DataType::Float16);
    EXPECT_EQ(parseDataType("f16"), DataType::Float16);
}

TEST_F(DataTypeTest, ParseBFloat16) {
    EXPECT_EQ(parseDataType("bfloat16"), DataType::BFloat16);
    EXPECT_EQ(parseDataType("bf16"), DataType::BFloat16);
}

TEST_F(DataTypeTest, ParseInt8) {
    EXPECT_EQ(parseDataType("int8"), DataType::Int8);
    EXPECT_EQ(parseDataType("i8"), DataType::Int8);
}

TEST_F(DataTypeTest, ParseUInt8) {
    EXPECT_EQ(parseDataType("uint8"), DataType::UInt8);
    EXPECT_EQ(parseDataType("u8"), DataType::UInt8);
}

TEST_F(DataTypeTest, ParseInt32) {
    EXPECT_EQ(parseDataType("int32"), DataType::Int32);
    EXPECT_EQ(parseDataType("int"), DataType::Int32);
    EXPECT_EQ(parseDataType("i32"), DataType::Int32);
}

TEST_F(DataTypeTest, ParseUInt32) {
    EXPECT_EQ(parseDataType("uint32"), DataType::UInt32);
    EXPECT_EQ(parseDataType("u32"), DataType::UInt32);
}

TEST_F(DataTypeTest, ParseInt64) {
    EXPECT_EQ(parseDataType("int64"), DataType::Int64);
    EXPECT_EQ(parseDataType("i64"), DataType::Int64);
}

TEST_F(DataTypeTest, ParseUInt64) {
    EXPECT_EQ(parseDataType("uint64"), DataType::UInt64);
    EXPECT_EQ(parseDataType("u64"), DataType::UInt64);
}

TEST_F(DataTypeTest, ParseInvalidReturnsDefault) {
    EXPECT_EQ(parseDataType(""), DataType::Float32);
    EXPECT_EQ(parseDataType("invalid"), DataType::Float32);
    EXPECT_EQ(parseDataType("FLOAT32"), DataType::Float32);  // Case sensitive
}

TEST_F(DataTypeTest, RoundTrip) {
    for (auto dtype : {DataType::Float32, DataType::Float64, DataType::Float16,
                       DataType::BFloat16, DataType::Int8, DataType::UInt8,
                       DataType::Int32, DataType::UInt32, DataType::Int64,
                       DataType::UInt64}) {
        EXPECT_EQ(parseDataType(dataTypeName(dtype)), dtype);
    }
}

// ============================================================================
// ReduceOp Tests
// ============================================================================

class ReduceOpTest : public ::testing::Test {};

TEST_F(ReduceOpTest, NameReturnsCorrectStrings) {
    EXPECT_STREQ(reduceOpName(ReduceOp::Sum), "sum");
    EXPECT_STREQ(reduceOpName(ReduceOp::Prod), "prod");
    EXPECT_STREQ(reduceOpName(ReduceOp::Min), "min");
    EXPECT_STREQ(reduceOpName(ReduceOp::Max), "max");
    EXPECT_STREQ(reduceOpName(ReduceOp::Avg), "avg");
}

TEST_F(ReduceOpTest, ParseSum) {
    EXPECT_EQ(parseReduceOp("sum"), ReduceOp::Sum);
}

TEST_F(ReduceOpTest, ParseProd) {
    EXPECT_EQ(parseReduceOp("prod"), ReduceOp::Prod);
}

TEST_F(ReduceOpTest, ParseMin) {
    EXPECT_EQ(parseReduceOp("min"), ReduceOp::Min);
}

TEST_F(ReduceOpTest, ParseMax) {
    EXPECT_EQ(parseReduceOp("max"), ReduceOp::Max);
}

TEST_F(ReduceOpTest, ParseAvg) {
    EXPECT_EQ(parseReduceOp("avg"), ReduceOp::Avg);
}

TEST_F(ReduceOpTest, ParseInvalidReturnsDefault) {
    EXPECT_EQ(parseReduceOp(""), ReduceOp::Sum);
    EXPECT_EQ(parseReduceOp("invalid"), ReduceOp::Sum);
    EXPECT_EQ(parseReduceOp("SUM"), ReduceOp::Sum);  // Case sensitive
}

TEST_F(ReduceOpTest, RoundTrip) {
    for (auto op : {ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Min,
                    ReduceOp::Max, ReduceOp::Avg}) {
        EXPECT_EQ(parseReduceOp(reduceOpName(op)), op);
    }
}

// ============================================================================
// Algorithm Tests
// ============================================================================

class AlgorithmTest : public ::testing::Test {};

TEST_F(AlgorithmTest, NameReturnsCorrectStrings) {
    EXPECT_STREQ(algorithmName(Algorithm::Auto), "auto");
    EXPECT_STREQ(algorithmName(Algorithm::Ring), "ring");
    EXPECT_STREQ(algorithmName(Algorithm::Tree), "tree");
    EXPECT_STREQ(algorithmName(Algorithm::CollNetDirect), "collnetdirect");
    EXPECT_STREQ(algorithmName(Algorithm::CollNetChain), "collnetchain");
    EXPECT_STREQ(algorithmName(Algorithm::NVLS), "nvls");
}

// ============================================================================
// Protocol Tests
// ============================================================================

class ProtocolTest : public ::testing::Test {};

TEST_F(ProtocolTest, NameReturnsCorrectStrings) {
    EXPECT_STREQ(protocolName(Protocol::Auto), "auto");
    EXPECT_STREQ(protocolName(Protocol::Simple), "simple");
    EXPECT_STREQ(protocolName(Protocol::LL), "ll");
    EXPECT_STREQ(protocolName(Protocol::LL128), "ll128");
}

// ============================================================================
// LinkType Tests
// ============================================================================

class LinkTypeTest : public ::testing::Test {};

TEST_F(LinkTypeTest, LegendReturnsCorrectStrings) {
    EXPECT_STREQ(linkTypeLegend(LinkType::Same), "X");
    EXPECT_STREQ(linkTypeLegend(LinkType::NVLink), "NV");
    EXPECT_STREQ(linkTypeLegend(LinkType::NVSwitch), "NVS");
    EXPECT_STREQ(linkTypeLegend(LinkType::C2C), "C2C");
    EXPECT_STREQ(linkTypeLegend(LinkType::PIX), "PIX");
    EXPECT_STREQ(linkTypeLegend(LinkType::PXB), "PXB");
    EXPECT_STREQ(linkTypeLegend(LinkType::PHB), "PHB");
    EXPECT_STREQ(linkTypeLegend(LinkType::NODE), "NODE");
    EXPECT_STREQ(linkTypeLegend(LinkType::SYS), "SYS");
    EXPECT_STREQ(linkTypeLegend(LinkType::NET), "NET");
}

// ============================================================================
// Struct Default Value Tests
// ============================================================================

class StructDefaultsTest : public ::testing::Test {};

TEST_F(StructDefaultsTest, NVLinkInfoDefaults) {
    NVLinkInfo info;
    EXPECT_EQ(info.sourceGpu, -1);
    EXPECT_EQ(info.targetGpu, -1);
    EXPECT_EQ(info.linkIndex, 0);
    EXPECT_EQ(info.version, 0);
    EXPECT_FALSE(info.isActive);
    EXPECT_TRUE(info.remotePciBusId.empty());
}

TEST_F(StructDefaultsTest, P2PInfoDefaults) {
    P2PInfo info;
    EXPECT_EQ(info.gpu1, 0);
    EXPECT_EQ(info.gpu2, 0);
    EXPECT_FALSE(info.accessSupported);
    EXPECT_FALSE(info.atomicSupported);
    EXPECT_EQ(info.performanceRank, 0);
    EXPECT_EQ(info.linkType, LinkType::SYS);
    EXPECT_EQ(info.nvlinkLanes, 0);
    EXPECT_EQ(info.nvlinkVersion, 0);
}

TEST_F(StructDefaultsTest, RDMAInfoDefaults) {
    RDMAInfo info;
    EXPECT_TRUE(info.deviceName.empty());
    EXPECT_TRUE(info.portState.empty());
    EXPECT_EQ(info.portNumber, 1);
    EXPECT_TRUE(info.linkType.empty());
    EXPECT_DOUBLE_EQ(info.rateGbps, 0.0);
    EXPECT_EQ(info.guid, 0u);
    EXPECT_TRUE(info.affinityGpus.empty());
    EXPECT_FALSE(info.gdrSupported);
}

TEST_F(StructDefaultsTest, GPUInfoDefaults) {
    GPUInfo info;
    EXPECT_EQ(info.deviceId, 0);
    EXPECT_TRUE(info.name.empty());
    EXPECT_TRUE(info.uuid.empty());
    EXPECT_TRUE(info.pciBusId.empty());
    EXPECT_EQ(info.computeCapabilityMajor, 0);
    EXPECT_EQ(info.computeCapabilityMinor, 0);
    EXPECT_EQ(info.totalMemoryBytes, 0u);
    EXPECT_EQ(info.numaNode, -1);
    EXPECT_EQ(info.nvlinkCount, 0);
    EXPECT_FALSE(info.gdrSupported);
    EXPECT_TRUE(info.nvlinks.empty());
}

TEST_F(StructDefaultsTest, TopologyInfoDefaults) {
    TopologyInfo info;
    EXPECT_TRUE(info.hostname.empty());
    EXPECT_EQ(info.ncclVersionMajor, 0);
    EXPECT_EQ(info.ncclVersionMinor, 0);
    EXPECT_EQ(info.ncclVersionPatch, 0);
    EXPECT_FALSE(info.hasNVSwitch);
    EXPECT_TRUE(info.gpus.empty());
    EXPECT_TRUE(info.p2pMatrix.empty());
    EXPECT_TRUE(info.rdmaDevices.empty());
}

}  // namespace testing
}  // namespace nperf

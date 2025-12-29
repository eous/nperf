/**
 * Unit tests for nperf/verification/verifier.h
 *
 * Tests verification logic that doesn't require GPU.
 * GPU-dependent tests are in integration/test_verifier.cpp
 */

#include <gtest/gtest.h>
#include "nperf/verification/verifier.h"
#include "test_utils.h"

namespace nperf {
namespace testing {

// ============================================================================
// getInitValue Tests
// ============================================================================

class GetInitValueTest : public ::testing::Test {};

TEST_F(GetInitValueTest, Rank0) {
    EXPECT_DOUBLE_EQ(getInitValue(0, 0), 1.0);
}

TEST_F(GetInitValueTest, Rank1) {
    EXPECT_DOUBLE_EQ(getInitValue(1, 0), 2.0);
}

TEST_F(GetInitValueTest, Rank7) {
    EXPECT_DOUBLE_EQ(getInitValue(7, 0), 8.0);
}

TEST_F(GetInitValueTest, IndexDoesNotAffectValue) {
    // Currently index is unused, but included for future extensibility
    EXPECT_DOUBLE_EQ(getInitValue(3, 0), getInitValue(3, 100));
    EXPECT_DOUBLE_EQ(getInitValue(3, 0), getInitValue(3, 1000000));
}

TEST_F(GetInitValueTest, PatternIsRankPlusOne) {
    for (int rank = 0; rank < 16; ++rank) {
        EXPECT_DOUBLE_EQ(getInitValue(rank, 0), static_cast<double>(rank + 1));
    }
}

// ============================================================================
// VerifyResult Default Values Tests
// ============================================================================

class VerifyResultDefaultsTest : public ::testing::Test {};

TEST_F(VerifyResultDefaultsTest, DefaultPassed) {
    VerifyResult result;
    EXPECT_TRUE(result.passed);
}

TEST_F(VerifyResultDefaultsTest, DefaultErrorCount) {
    VerifyResult result;
    EXPECT_EQ(result.errorCount, 0);
}

TEST_F(VerifyResultDefaultsTest, DefaultFirstErrorIndex) {
    VerifyResult result;
    EXPECT_EQ(result.firstErrorIndex, 0u);
}

TEST_F(VerifyResultDefaultsTest, DefaultExpectedValue) {
    VerifyResult result;
    EXPECT_DOUBLE_EQ(result.expectedValue, 0.0);
}

TEST_F(VerifyResultDefaultsTest, DefaultActualValue) {
    VerifyResult result;
    EXPECT_DOUBLE_EQ(result.actualValue, 0.0);
}

TEST_F(VerifyResultDefaultsTest, DefaultMessage) {
    VerifyResult result;
    EXPECT_TRUE(result.message.empty());
}

// ============================================================================
// Verifier Expected Value Tests
// ============================================================================

class VerifierExpectedValueTest : public ::testing::Test {};

TEST_F(VerifierExpectedValueTest, AllReduceSum2Ranks) {
    Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, 2, 0);

    // AllReduce sum: sum(rank+1 for rank in 0..worldSize-1)
    // For 2 ranks: 1 + 2 = 3
    double expected = verifier.getExpectedValue(0);
    EXPECT_DOUBLE_EQ(expected, 3.0);
}

TEST_F(VerifierExpectedValueTest, AllReduceSum4Ranks) {
    Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, 4, 0);

    // For 4 ranks: 1 + 2 + 3 + 4 = 10
    double expected = verifier.getExpectedValue(0);
    EXPECT_DOUBLE_EQ(expected, 10.0);
}

TEST_F(VerifierExpectedValueTest, AllReduceSum8Ranks) {
    Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, 8, 0);

    // For 8 ranks: 1 + 2 + ... + 8 = 36
    // Formula: n*(n+1)/2 = 8*9/2 = 36
    double expected = verifier.getExpectedValue(0);
    EXPECT_DOUBLE_EQ(expected, 36.0);
}

TEST_F(VerifierExpectedValueTest, AllReduceSumFormula) {
    // Verify the formula: worldSize*(worldSize+1)/2
    for (int worldSize = 1; worldSize <= 16; ++worldSize) {
        Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, worldSize, 0);
        double expected = verifier.getExpectedValue(0);
        double formula = static_cast<double>(worldSize * (worldSize + 1)) / 2.0;
        EXPECT_DOUBLE_EQ(expected, formula);
    }
}

TEST_F(VerifierExpectedValueTest, BroadcastFromRoot0) {
    Verifier verifier(CollectiveOp::Broadcast, DataType::Float32, 4, 0);

    // Broadcast from root 0: all ranks receive root's value (root+1 = 1)
    double expected = verifier.getExpectedValue(0);
    EXPECT_DOUBLE_EQ(expected, 1.0);
}

TEST_F(VerifierExpectedValueTest, BroadcastOnNonRootRank) {
    // Verifier on rank 2, but broadcast is from root 0
    Verifier verifier(CollectiveOp::Broadcast, DataType::Float32, 4, 2);

    // All ranks receive root's value
    double expected = verifier.getExpectedValue(0);
    EXPECT_DOUBLE_EQ(expected, 1.0);  // Root is assumed to be 0
}

TEST_F(VerifierExpectedValueTest, ReduceOnRoot) {
    // Reduce to root (rank 0)
    Verifier verifier(CollectiveOp::Reduce, DataType::Float32, 4, 0);

    // Root receives the reduction result: sum = 1+2+3+4 = 10
    double expected = verifier.getExpectedValue(0);
    EXPECT_DOUBLE_EQ(expected, 10.0);
}

TEST_F(VerifierExpectedValueTest, ReduceOnNonRoot) {
    // Non-root rank (rank 2)
    Verifier verifier(CollectiveOp::Reduce, DataType::Float32, 4, 2);

    // Non-root ranks keep their original value
    double expected = verifier.getExpectedValue(0);
    double initValue = getInitValue(2, 0);  // rank+1 = 3
    EXPECT_DOUBLE_EQ(expected, initValue);
}

TEST_F(VerifierExpectedValueTest, ReduceScatter) {
    Verifier verifier(CollectiveOp::ReduceScatter, DataType::Float32, 4, 0);

    // ReduceScatter: similar to AllReduce, each rank gets part of reduced result
    double expected = verifier.getExpectedValue(0);
    EXPECT_DOUBLE_EQ(expected, 10.0);  // Same as AllReduce sum
}

TEST_F(VerifierExpectedValueTest, AllGather) {
    // AllGather verification is position-dependent in general
    // Current implementation just verifies value is in valid range
    Verifier verifier(CollectiveOp::AllGather, DataType::Float32, 4, 1);

    double expected = verifier.getExpectedValue(0);
    double initValue = getInitValue(1, 0);
    // For AllGather, expected returns init value (simplified verification)
    EXPECT_DOUBLE_EQ(expected, initValue);
}

// ============================================================================
// Verifier Tolerance Tests
// ============================================================================

// We can't easily test the compare function since it's private,
// but we can verify tolerance is set correctly
class VerifierToleranceTest : public ::testing::Test {};

TEST_F(VerifierToleranceTest, DefaultTolerance) {
    // Default tolerance is 1e-5
    Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, 2, 0);
    // Can't access tolerance_ directly, but verifier should work with default
    SUCCEED();  // Construction doesn't throw
}

TEST_F(VerifierToleranceTest, SetCustomTolerance) {
    Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, 2, 0);
    verifier.setTolerance(1e-3);
    SUCCEED();  // No exception
}

TEST_F(VerifierToleranceTest, SetTightTolerance) {
    Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, 2, 0);
    verifier.setTolerance(1e-10);
    SUCCEED();  // No exception
}

TEST_F(VerifierToleranceTest, SetLooseTolerance) {
    Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, 2, 0);
    verifier.setTolerance(0.1);  // 10%
    SUCCEED();  // No exception
}

// ============================================================================
// Verifier Construction Tests
// ============================================================================

class VerifierConstructionTest : public ::testing::Test {};

TEST_F(VerifierConstructionTest, AllCollectiveOps) {
    // Verify construction works for all operations
    for (auto op : {CollectiveOp::AllReduce, CollectiveOp::AllGather,
                    CollectiveOp::Broadcast, CollectiveOp::Reduce,
                    CollectiveOp::ReduceScatter, CollectiveOp::AlltoAll,
                    CollectiveOp::Gather, CollectiveOp::Scatter,
                    CollectiveOp::SendRecv}) {
        Verifier verifier(op, DataType::Float32, 4, 0);
        SUCCEED();  // Construction doesn't throw
    }
}

TEST_F(VerifierConstructionTest, AllDataTypes) {
    // Verify construction works for all data types
    for (auto dtype : {DataType::Float32, DataType::Float64,
                       DataType::Float16, DataType::BFloat16,
                       DataType::Int8, DataType::UInt8,
                       DataType::Int32, DataType::UInt32,
                       DataType::Int64, DataType::UInt64}) {
        Verifier verifier(CollectiveOp::AllReduce, dtype, 4, 0);
        SUCCEED();  // Construction doesn't throw
    }
}

TEST_F(VerifierConstructionTest, VariousWorldSizes) {
    for (int worldSize = 1; worldSize <= 128; worldSize *= 2) {
        Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, worldSize, 0);
        SUCCEED();  // Construction doesn't throw
    }
}

TEST_F(VerifierConstructionTest, VariousRanks) {
    int worldSize = 8;
    for (int rank = 0; rank < worldSize; ++rank) {
        Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, worldSize, rank);
        SUCCEED();  // Construction doesn't throw
    }
}

// ============================================================================
// Expected Value Consistency Tests
// ============================================================================

class VerifierConsistencyTest : public ::testing::Test {};

TEST_F(VerifierConsistencyTest, AllReduceSameAcrossRanks) {
    // For AllReduce, all ranks should have the same expected value
    int worldSize = 8;
    double firstExpected = -1.0;

    for (int rank = 0; rank < worldSize; ++rank) {
        Verifier verifier(CollectiveOp::AllReduce, DataType::Float32, worldSize, rank);
        double expected = verifier.getExpectedValue(0);

        if (rank == 0) {
            firstExpected = expected;
        } else {
            EXPECT_DOUBLE_EQ(expected, firstExpected);
        }
    }
}

TEST_F(VerifierConsistencyTest, BroadcastSameAcrossRanks) {
    // For Broadcast, all ranks should have the same expected value
    int worldSize = 8;
    double firstExpected = -1.0;

    for (int rank = 0; rank < worldSize; ++rank) {
        Verifier verifier(CollectiveOp::Broadcast, DataType::Float32, worldSize, rank);
        double expected = verifier.getExpectedValue(0);

        if (rank == 0) {
            firstExpected = expected;
        } else {
            EXPECT_DOUBLE_EQ(expected, firstExpected);
        }
    }
}

TEST_F(VerifierConsistencyTest, ReduceRootIsDifferent) {
    // For Reduce, root should have reduction result, others keep init
    int worldSize = 4;
    int root = 0;

    Verifier rootVerifier(CollectiveOp::Reduce, DataType::Float32, worldSize, root);
    double rootExpected = rootVerifier.getExpectedValue(0);

    // Root should have sum
    EXPECT_DOUBLE_EQ(rootExpected, 10.0);  // 1+2+3+4

    // Non-root should have init value
    for (int rank = 1; rank < worldSize; ++rank) {
        Verifier verifier(CollectiveOp::Reduce, DataType::Float32, worldSize, rank);
        double expected = verifier.getExpectedValue(0);
        double initValue = getInitValue(rank, 0);
        EXPECT_DOUBLE_EQ(expected, initValue);
    }
}

}  // namespace testing
}  // namespace nperf

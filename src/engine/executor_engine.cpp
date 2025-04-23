#include "engine/executor_engine.hpp"
#include "engine/operators/list.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include <chrono>

void ExecutorEngine::execute(duckdb::LogicalOperator& op, const duckdb::Planner& planner) {
    CacheManager cacheManager;
    // resolve to actual data
    ColumnBindingResolver resolver;
    resolver.VisitOperator(op);
    // Initialize CSV writer with a timestamped file name
    auto now = std::chrono::system_clock::now();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::string csvPath = YallaSQL::resultDir + "/our_output_" + std::to_string(millis) + ".csv";
    CsvWriter csvWriter(csvPath);

    auto rootOperator = YallaSQL::Operator::CreateOperator(op, planner);
    if (!rootOperator) {
        std::cerr << "Failed to create operator\n";
        return;
    }
    rootOperator->init();

    BatchID batchId = rootOperator->next(cacheManager);
    while (batchId != 0) {
        std::unique_ptr<Batch> batch = cacheManager.getBatch(batchId);
        if (!batch) {
            std::cerr << "Failed to get batch\n";
            return;
        }
        batch->moveTo(Device::CPU); // Ensure batch is on CPU for CSV writing
        csvWriter.addBatch(*batch);
//        std::cout << "Processed batch ID: " << batchId << "\n";
        batchId = rootOperator->next(cacheManager);
    }
}
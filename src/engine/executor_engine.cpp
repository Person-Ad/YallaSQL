#include "engine/executor_engine.hpp"
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

    std::vector<BatchID> buffer_ids;

    BatchID batchId = rootOperator->next(cacheManager);
    while (batchId != 0) {
        buffer_ids.push_back(batchId);
        if(buffer_ids.size() > 50) 
            saveBufferedBatchs(buffer_ids, cacheManager, csvWriter);

        batchId = rootOperator->next(cacheManager);
    }
    saveBufferedBatchs(buffer_ids, cacheManager, csvWriter);
}

void ExecutorEngine::saveBufferedBatchs(std::vector<BatchID>& buffer_ids, CacheManager &cacheManager, CsvWriter &csvWriter) {
    std::vector<std::unique_ptr<Batch>> buffer;
    buffer.reserve(buffer_ids.size());
    for(auto id: buffer_ids) {
        buffer.push_back( cacheManager.getBatch(id)  );
    }
    // Ensure batch is on CPU for CSV writing 
    // send multiple request Async
    for(auto& batch: buffer) {
        batch->moveTo(Device::CPU);
    }
    // write and ensure to sync
    for(auto& batch: buffer) {
        CUDA_CHECK(cudaStreamSynchronize(batch->stream));
        CUDA_CHECK(cudaStreamDestroy(batch->stream));
        csvWriter.addBatch(*batch);
    }

    buffer_ids.clear();
}

#include "engine/executor_engine.hpp"
#include "engine/operators/list.hpp"

void ExecutorEngine::execute(const duckdb::LogicalOperator& op, const duckdb::Planner& planner) {
    CacheManager cacheManager;

    auto rootOperator = YallaSQL::Operator::CreateOperator(op, planner);
    if(!rootOperator) {
        std::cerr << "Failed to create operator\n";
        return;
    }
    rootOperator->init();

    BatchID batchId = rootOperator->next(cacheManager);
    std::unique_ptr<Batch> batch;
    int total = 0;
    while (batchId != 0) {
        batch = cacheManager.getBatch(batchId);

        if(!batch) {
            std::cerr << "Failed to get batch\n";
            return;
        }
        total += batch->batchSize;
        std::cout << "Finish Execution of : " << batchId << " : " << total << " rows\n";
        batchId = rootOperator->next(cacheManager);
    }
    // batch->moveTo(Device::FS);
    // batch->print();

    // if(opType == LogicalOperatorType::LOGICAL_PROJECTION) {
    //     YallaSQL::ProjectionOperator ourOP(op);
    //     ourOP.init();
    //     BatchID batchId = ourOP.next(cacheManager);
    //     // Batch* batch = cacheManager.getBatch(batchId, Device::CPU);
    //     // // batch->print();
    //     // batch->moveTo(Device::GPU);
    //     // batch->moveTo(Device::FS);
    //     // batch->print();
    // } 
    // else if (opType == LogicalOperatorType::LOGICAL_GET) {
    //     // const LogicalGet& castOp = op.Cast<LogicalGet>();
    //     // YallaSQL::GetOperator ourOP(op);
    //     // ourOP.init();
    //     // BatchID batchId = ourOP.next(cacheManager);
    //     // Batch* batch = cacheManager.getBatch(batchId, Device::CPU);
    //     // // batch->print();

    //     // batch->moveTo(Device::GPU);
    //     // batch->moveTo(Device::FS);
    //     // batch->moveTo(Device::CPU);
    //     // batch->print();

    // }

    // for(int i = 0;i < numExpressions;i++) {
    //     auto expType = op.expressions[i]->type;
    //     if(expType == ExpressionType::BOUND_COLUMN_REF) {
    //         auto &castExpr = op.expressions[i]->Cast<BoundColumnRefExpression>();
    //         duckdb::idx_t col = castExpr.binding.column_index;
    //         duckdb::idx_t table = castExpr.binding.table_index;
    //         std::cout << col << " " << table << "\n";
    //     }
    //     // duckdb::ExpressionIterator::EnumerateExpression()
    // }
    // if(op.children.size() == 0) return;
    // for(int i = 0;i < numChild;i++){
    //     if(op.children[i])
    //         getPhysicalPlan(*op.children[i], cacheManager);
    // }
        
}
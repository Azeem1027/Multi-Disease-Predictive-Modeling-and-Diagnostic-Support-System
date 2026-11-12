"""
Master Training Script to Train All Four Disease Models and Baselines End-to-End.
Generates all models, metadata JSONs, ROC curves, confusion matrices, and SHAP summaries.
"""
import time
import logging
from src.train_liver import train_liver
from src.train_kidney import train_kidney
from src.train_diabetes import train_diabetes
from src.train_heart import train_heart

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
#logging in save the track and history and print on a single go rather than using multiple statement

def main():
    start_total = time.time()
    logger.info("=================================================================")
    logger.info("STARTING COMPLETE MULTI-DISEASE MODEL TRAINING PIPELINE")
    logger.info("=================================================================")

    results = {}  #initialize an empty dictionary to store the matrix results

    # 1. Liver Disease
    t0 = time.time()
    results["liver"] = train_liver()
    logger.info("Liver training finished in %.1f seconds", time.time() - t0)

    # 2. Kidney Disease
    t0 = time.time()
    results["kidney"] = train_kidney()
    logger.info("Kidney training finished in %.1f seconds", time.time() - t0)

    # 3. Diabetes
    t0 = time.time()
    results["diabetes"] = train_diabetes()
    logger.info("Diabetes training finished in %.1f seconds", time.time() - t0)

    # 4. Heart Disease
    t0 = time.time()
    results["heart"] = train_heart()
    logger.info("Heart training finished in %.1f seconds", time.time() - t0)

    total_time = time.time() - start_total
    logger.info("=================================================================")
    logger.info("ALL FOUR DISEASE MODELS TRAINED AND SAVED IN %.1f SECONDS!", total_time)
    logger.info("=================================================================")

    # Print summary table
    print("\n" + "=" * 92)
    print(f"{'Disease':<12} | {'Model Family':<26} | {'Accuracy':<8} | {'Recall':<8} | {'Precision':<9} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 92)
    for disease, meta in results.items():
        m = meta["tuned_metrics"]
        print(f"{disease.capitalize():<12} | {meta['primary_model']:<26} | {m['accuracy']:<8.4f} | {m['recall']:<8.4f} | {m['precision']:<9.4f} | {m['f1']:<6.4f} | {m['roc_auc']:<7.4f}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()


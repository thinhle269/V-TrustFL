import config
import data_processor
import engine

def main():
    print("="*65)
    print(" BẮT ĐẦU TRAINING THỰC TẾ: SCIE Q1 PROJECT")
    print("="*65)
    config.setup_env()
    data_processor.prepare_data()
    engine.run_baselines()

if __name__ == "__main__": main()

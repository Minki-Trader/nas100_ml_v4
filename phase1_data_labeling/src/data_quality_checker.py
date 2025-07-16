"""
Phase 1.1: 데이터 품질 재검증
- 시간 갭 확인
- 이상치 제거 (일일 20% 이상 변동 등)
- 스프레드 데이터 정상 범위 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataQualityChecker:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.quality_report = {}
        
    def load_data(self):
        """데이터 로드 및 기본 전처리"""
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # 시간 컬럼 확인 및 변환
        time_columns = ['Time', 'time', 'datetime', 'Datetime']
        time_col = None
        for col in time_columns:
            if col in self.df.columns:
                time_col = col
                break
        
        if time_col:
            self.df['datetime'] = pd.to_datetime(self.df[time_col])
            self.df.set_index('datetime', inplace=True)
            self.df.sort_index(inplace=True)
        
        print(f"Data shape: {self.df.shape}")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def check_time_gaps(self, expected_freq='15T'):
        """시간 갭 확인"""
        print("\n=== Checking Time Gaps ===")
        
        # 시간 차이 계산
        time_diff = self.df.index.to_series().diff()
        
        # 예상 주기
        expected_delta = pd.Timedelta(expected_freq)
        
        # 갭 찾기 (예상보다 큰 경우)
        gaps = time_diff[time_diff > expected_delta * 1.5]
        
        if len(gaps) > 0:
            print(f"Found {len(gaps)} time gaps:")
            gap_info = []
            for idx, gap in gaps.items():
                gap_info.append({
                    'time': idx,
                    'gap_duration': gap,
                    'previous_time': idx - gap
                })
            
            gap_df = pd.DataFrame(gap_info)
            print(gap_df.head(10))
            
            # 주말 갭 vs 비정상 갭 구분
            weekend_gaps = []
            abnormal_gaps = []
            
            for _, row in gap_df.iterrows():
                prev_time = pd.Timestamp(row['previous_time'])
                curr_time = pd.Timestamp(row['time'])
                
                # 금요일 -> 일요일/월요일 갭인지 확인
                if prev_time.weekday() == 4 and curr_time.weekday() in [6, 0]:
                    weekend_gaps.append(row)
                else:
                    abnormal_gaps.append(row)
            
            print(f"\nWeekend gaps: {len(weekend_gaps)}")
            print(f"Abnormal gaps: {len(abnormal_gaps)}")
            
            if abnormal_gaps:
                print("\nAbnormal gaps (non-weekend):")
                print(pd.DataFrame(abnormal_gaps).head())
            
            self.quality_report['time_gaps'] = {
                'total_gaps': len(gaps),
                'weekend_gaps': len(weekend_gaps),
                'abnormal_gaps': len(abnormal_gaps),
                'gap_details': gap_df.to_dict()
            }
        else:
            print("No significant time gaps found")
            self.quality_report['time_gaps'] = {'total_gaps': 0}
    
    def check_outliers(self, threshold=0.2):
        """이상치 확인 (일일 20% 이상 변동)"""
        print("\n=== Checking Outliers ===")
        
        # 일일 수익률 계산
        self.df['returns'] = self.df['Close'].pct_change()
        
        # 극단적 변동 찾기
        extreme_moves = self.df[abs(self.df['returns']) > threshold]
        
        if len(extreme_moves) > 0:
            print(f"Found {len(extreme_moves)} extreme moves (>{threshold*100}%):")
            print(extreme_moves[['Open', 'High', 'Low', 'Close', 'returns']].head(10))
            
            # 통계
            print(f"\nMax positive move: {self.df['returns'].max():.2%}")
            print(f"Max negative move: {self.df['returns'].min():.2%}")
            
            self.quality_report['outliers'] = {
                'extreme_moves_count': len(extreme_moves),
                'max_positive': float(self.df['returns'].max()),
                'max_negative': float(self.df['returns'].min()),
                'threshold': threshold
            }
        else:
            print(f"No extreme moves found (>{threshold*100}%)")
            self.quality_report['outliers'] = {'extreme_moves_count': 0}
        
        # 수익률 분포 시각화
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        self.df['returns'].hist(bins=100, alpha=0.7)
        plt.axvline(-threshold, color='r', linestyle='--', label=f'-{threshold*100}%')
        plt.axvline(threshold, color='r', linestyle='--', label=f'+{threshold*100}%')
        plt.title('Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        self.df['returns'].plot(alpha=0.5)
        plt.axhline(threshold, color='r', linestyle='--', alpha=0.5)
        plt.axhline(-threshold, color='r', linestyle='--', alpha=0.5)
        plt.title('Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        
        plt.tight_layout()
        plt.savefig('returns_analysis.png')
        print("\nReturns analysis saved to 'returns_analysis.png'")
    
    def check_spread(self):
        """스프레드 데이터 확인"""
        print("\n=== Checking Spread Data ===")
        
        if 'Spread' in self.df.columns:
            spread_stats = self.df['Spread'].describe()
            print("Spread statistics:")
            print(spread_stats)
            
            # 스프레드가 0인 경우 확인
            zero_spread = self.df[self.df['Spread'] == 0]
            print(f"\nZero spread occurrences: {len(zero_spread)}")
            
            # 비정상적으로 큰 스프레드 확인
            spread_mean = self.df['Spread'].mean()
            spread_std = self.df['Spread'].std()
            abnormal_spread = self.df[self.df['Spread'] > spread_mean + 5 * spread_std]
            print(f"Abnormal spread occurrences (>5 std): {len(abnormal_spread)}")
            
            if len(abnormal_spread) > 0:
                print("\nAbnormal spread samples:")
                print(abnormal_spread[['Open', 'Close', 'Spread']].head())
            
            # 스프레드 시각화
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            self.df['Spread'].hist(bins=50, alpha=0.7)
            plt.axvline(spread_mean, color='r', linestyle='--', label=f'Mean: {spread_mean:.2f}')
            plt.axvline(spread_mean + 5*spread_std, color='orange', linestyle='--', 
                       label=f'5 STD: {spread_mean + 5*spread_std:.2f}')
            plt.title('Spread Distribution')
            plt.xlabel('Spread')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            self.df['Spread'].resample('D').mean().plot()
            plt.title('Daily Average Spread')
            plt.xlabel('Date')
            plt.ylabel('Average Spread')
            
            plt.tight_layout()
            plt.savefig('spread_analysis.png')
            print("\nSpread analysis saved to 'spread_analysis.png'")
            
            self.quality_report['spread'] = {
                'mean': float(spread_mean),
                'std': float(spread_std),
                'zero_spread_count': len(zero_spread),
                'abnormal_spread_count': len(abnormal_spread)
            }
        else:
            print("No 'Spread' column found in data")
            self.quality_report['spread'] = {'status': 'No spread data'}
    
    def check_data_integrity(self):
        """데이터 무결성 확인"""
        print("\n=== Checking Data Integrity ===")
        
        # OHLC 관계 확인
        ohlc_errors = []
        
        # High >= Low
        high_low_errors = self.df[self.df['High'] < self.df['Low']]
        if len(high_low_errors) > 0:
            ohlc_errors.append(f"High < Low: {len(high_low_errors)} cases")
        
        # High >= Open, Close
        high_open_errors = self.df[self.df['High'] < self.df['Open']]
        high_close_errors = self.df[self.df['High'] < self.df['Close']]
        if len(high_open_errors) > 0:
            ohlc_errors.append(f"High < Open: {len(high_open_errors)} cases")
        if len(high_close_errors) > 0:
            ohlc_errors.append(f"High < Close: {len(high_close_errors)} cases")
        
        # Low <= Open, Close
        low_open_errors = self.df[self.df['Low'] > self.df['Open']]
        low_close_errors = self.df[self.df['Low'] > self.df['Close']]
        if len(low_open_errors) > 0:
            ohlc_errors.append(f"Low > Open: {len(low_open_errors)} cases")
        if len(low_close_errors) > 0:
            ohlc_errors.append(f"Low > Close: {len(low_close_errors)} cases")
        
        if ohlc_errors:
            print("OHLC relationship errors found:")
            for error in ohlc_errors:
                print(f"  - {error}")
        else:
            print("All OHLC relationships are valid")
        
        # 결측치 확인
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            print("\nMissing values found:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found")
        
        # 중복 인덱스 확인
        duplicate_indices = self.df.index.duplicated().sum()
        if duplicate_indices > 0:
            print(f"\nDuplicate indices found: {duplicate_indices}")
        else:
            print("\nNo duplicate indices found")
        
        self.quality_report['integrity'] = {
            'ohlc_errors': ohlc_errors,
            'missing_values': missing_values.to_dict(),
            'duplicate_indices': int(duplicate_indices)
        }
    
    def generate_report(self):
        """품질 검사 보고서 생성"""
        print("\n=== Quality Check Summary ===")
        
        # 시간 갭 요약
        if 'time_gaps' in self.quality_report:
            gaps = self.quality_report['time_gaps']
            print(f"\nTime Gaps: {gaps.get('total_gaps', 0)} total "
                  f"({gaps.get('weekend_gaps', 0)} weekend, "
                  f"{gaps.get('abnormal_gaps', 0)} abnormal)")
        
        # 이상치 요약
        if 'outliers' in self.quality_report:
            outliers = self.quality_report['outliers']
            print(f"\nOutliers: {outliers.get('extreme_moves_count', 0)} extreme moves")
            print(f"  Max positive: {outliers.get('max_positive', 0):.2%}")
            print(f"  Max negative: {outliers.get('max_negative', 0):.2%}")
        
        # 스프레드 요약
        if 'spread' in self.quality_report:
            spread = self.quality_report['spread']
            if 'mean' in spread:
                print(f"\nSpread: Mean={spread['mean']:.2f}, STD={spread['std']:.2f}")
                print(f"  Zero spread: {spread.get('zero_spread_count', 0)} cases")
                print(f"  Abnormal spread: {spread.get('abnormal_spread_count', 0)} cases")
        
        # 무결성 요약
        if 'integrity' in self.quality_report:
            integrity = self.quality_report['integrity']
            print(f"\nData Integrity:")
            if integrity['ohlc_errors']:
                for error in integrity['ohlc_errors']:
                    print(f"  - {error}")
            else:
                print("  - All OHLC relationships valid")
            print(f"  - Duplicate indices: {integrity['duplicate_indices']}")
        
        # 보고서 저장
        import json
        report_path = 'data_quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, indent=4, default=str)
        print(f"\nDetailed report saved to '{report_path}'")
        
        return self.quality_report

# 사용 예시
if __name__ == "__main__":
    # 데이터 경로 설정
    data_path = "../data/raw/NAS100_ESSENTIAL_V2_M15.csv"
    
    # 품질 검사 실행
    checker = DataQualityChecker(data_path)
    
    # 데이터 로드
    df = checker.load_data()
    
    # 각 검사 수행
    checker.check_time_gaps(expected_freq='15T')  # 15분봉 기준
    checker.check_outliers(threshold=0.2)  # 20% 이상 변동
    checker.check_spread()
    checker.check_data_integrity()
    
    # 최종 보고서 생성
    report = checker.generate_report()

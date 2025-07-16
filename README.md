# NAS100 ML 거래 시스템 개발 로드맵 v4.0

## 🛡️ 동적 리스크 관리 시스템
생존확률 90% 이상을 유지하며 지속가능한 성장을 추구하는 적응형 ML 거래 시스템

## ⚠️ 중요 경고 및 면책조항

### 투자 위험 경고
- **이 시스템은 높은 금융 리스크를 포함합니다**
- **전체 투자금 손실 가능성이 있습니다**
- **레버리지 사용은 손실을 증폭시킬 수 있습니다**
- **과거 성과가 미래 수익을 보장하지 않습니다**
- **실제 거래 전 충분한 테스트와 검증이 필수입니다**

### 동적 리스크 전략 경고
이 로드맵은 **리스크%와 레버리지를 동적으로 최적화**하는 전략입니다.
- 생존확률 90% 이상 유지하면서 최대 수익 추구
- Walk-Forward마다 최적 리스크-레버리지 조합 재계산
- 시장 상태에 따라 일일 손실 한도 5-25% 변동
- 목표: 높은 생존확률로 지속가능한 성장

## 📋 v4.0 개선사항
- 2단계 리스크-레버리지 최적화 (MC-Kelly → Bayesian)
- Wilson interval 기반 RoR 계산
- EWMA spread (50틱) × 1.8 기준 더블 레짐
- Walk-Forward 설계 개선 (6M-1M-1M, 샘플수 보장)
- 롤오버 gap 및 스프레드 확대 반영
- 생존확률 기반 의사결정 (90% 타겟)
- Volatility regime별 동적 임계값
- PCA preview로 피처 군집 확인
- Bidirectional LSTM + Attention 옵션
- CatBoost 대안 모델 추가
- Quantile-based 진입 규칙
- 슬리피지 랜덤화 백테스트
- 실시간 데이터 지연 감지

## 🔑 핵심 보완 (P1 Priority)
1. **2단계 동적 리스크-레버리지 최적화**
   - MC-Kelly로 Risk 3개 후보 선별 (Low/Mid/High)
   - Bayesian optimization으로 레버리지 연속 탐색
   - 8-10회 평가로 수렴 (기존 50-60회 → 대폭 단축)
   
2. **Wilson interval 기반 RoR 계산**
   - 소표본 편향 완화
   - MC 입력: p(win), R:R, Risk%, #trades (4개 고정)
   - RoR < 10% 목표, > 30% 시 거래 중단
   
3. **더블 레짐 모델 (ATR × Spread)**
   - EWMA spread (50틱/5분) × 1.8 = High Spread
   - 15초마다 업데이트, 진입 직전 재확인
   - High-vol + High-spread = 진입 금지
   
4. **데이터 품질 강화**
   - Walk-Forward: 6M train → 1M val → 1M test
   - 최소 샘플수 보장 (train ≥ 20k, val/test ≥ 5k)
   - 롤오버 gap 반영 및 전후 3일 spread × 1.3

## 🎯 목표
52-55% 예측 정확도로 지속가능한 성장을 달성하는 동적 ML 시스템
- 생존확률 목표: > 90%
- Sharpe Ratio: > 0.8
- 리스크-수익 최적화: Walk-Forward마다 재계산
- 시장 적응형 포지션 사이징

---

## Phase 1: 데이터 검증 및 라벨링 재설계

### 1.1 데이터 품질 재검증
```python
# 기존 데이터 파일 체크
- 시간 갭 확인
- 이상치 제거 (일일 20% 이상 변동 등)
- 스프레드 데이터 정상 범위 확인
```

### 1.2 Triple Barrier 라벨링 구현
```python
# 동적 임계값 설정
- Primary: ATR 기반 threshold = 0.7 * ATR(20)
- Secondary: 고정 threshold = 0.002 (0.2%) 비교
- Volatility regime별 조정:
  * Low vol (VIX<15): 0.5 * ATR
  * Normal (VIX 15-25): 0.7 * ATR  
  * High vol (VIX>25): 1.0 * ATR
- 시간 한계: 4봉(1시간), 8봉(2시간) 테스트
- 중립 케이스는 학습에서 제외

# 구현 세부사항
- TP/SL 비대칭 테스트 
  * 보수적: TP=1.5*ATR, SL=1.0*ATR
  * 균형: TP=2.0*ATR, SL=1.0*ATR
  * 공격적: TP=2.5*ATR, SL=1.0*ATR
- 시간대별 임계값 조정 가능성 검토
- 주말/휴일 경계 처리
- 극단적 변동성 시 임계값 확대

# 예상 결과
- 학습 샘플 수: 기존 대비 60-70%
- 클래스 분포: 45:55 ~ 55:45 (균형)
- 일평균 라벨: 최적화 결과에 따라 변동
```

### 1.3 타겟 검증
```python
# 라벨 분포 분석
- 시간대별 승률 히트맵
- 요일별 패턴 분석
- 변동성 구간별 라벨 분포
- Volatility regime별 승률 차이
- 연속 승/패 분석
```

---

## Phase 2: 하이브리드 피처 엔지니어링

### 2.1 코어 피처 선정 (17개 고정)
```python
# 검증된 필수 피처 - 무조건 포함
1. 모멘텀 (3개)
   - RSI(14)
   - ROC(10) 
   - MACD_signal
   
2. 변동성 (4개)
   - ATR_ratio (current/avg)
   - realized_vol_20
   - BB_width
   - volatility_regime (low/normal/high)
   
3. 미시구조 (4개)
   - spread_ratio
   - spread_change_rate
   - high_low_ratio
   - spread_regime (high/normal)
   
4. 가격 액션 (3개)
   - price_position_in_range
   - MA20_deviation
   - candle_pattern_score
   
5. 시간 (3개)
   - hour_sin/cos
   - session_progress
   - high_volatility_hour flag
   
6. MTF 확인 (2개)
   - H1_trend_direction
   - H4_support_resistance_distance
   
7. 추가 (1개)
   - consecutive_moves (연속 상승/하락 카운트)
```

### 2.2 실험 피처 풀 (30개)
```python
# 통계적 선택 대상
1. 추가 모멘텀 지표
   - Stochastic, Williams %R, CCI
   - 다양한 기간의 ROC
   
2. 복합 지표
   - RSI 다이버전스
   - MACD 히스토그램 변화율
   
3. 가격 구조
   - 일중 고저 대비 현재 위치
   - 피벗 포인트 거리
   - 이전 고점/저점 돌파
   - Donchian 채널 위치
   
4. 패턴 인식
   - 캔들 패턴 variations
   - 차트 패턴 점수
   - 프랙탈 패턴
   
5. 시장 체제
   - ADX variations
   - Choppiness Index
   - Hurst Exponent (추세 강도)
   
6. 통계적 피처
   - 수익률 왜도/첨도
   - 자기상관 지표
   - 엔트로피 측정
   - 시간대별 평균 변동성 대비 현재
```

### 2.3 피처 선택 파이프라인
```python
# Step 1: 상관관계 필터 (|r| > 0.95 제거)
# Step 2: Variance Inflation Factor (VIF < 10)
# Step 3: PCA Preview (95% variance 확인)
# Step 4: Mutual Information 점수
# Step 5: Permutation Importance
# Step 6: 최종 20-25개 선택

# 검증
- 각 단계별 제거 피처 기록
- PCA 결과로 피처 군집 확인
- 최종 피처의 카테고리별 균형 확인
```

### 2.4 피처 전처리
```python
# 스케일링 전략
- 가격 기반: RobustScaler
- 지표 기반: MinMaxScaler(-1, 1)
- 카테고리: One-hot or Ordinal
- 시간: Cyclical encoding

# 결측치 처리
- 지표별 forward fill 한계 설정
- 초기 결측치 처리 전략

# 데이터 누수 방지 테스트
- 각 피처 생성 후 자동 검증
  ```python
  assert feature.shift(1).equals(feature_lagged)
  assert not feature.isna().all()
  assert feature.index.is_monotonic_increasing
  ```
- Look-ahead bias 탐지 유닛 테스트
- 피처별 타임스탬프 일관성 체크
```

---

## Phase 3: 모델 개발

### 3.1 베이스라인 모델
```python
# 단순 로지스틱 회귀
- 코어 피처만 사용
- 성능 하한선 설정 (목표: 51%+)
```

### 3.2 LSTM 주력 모델
```python
# 아키텍처
- Input: 30-50 timesteps
- LSTM layers: 2개 (64, 32 units)
  * Option: Bidirectional LSTM
  * Option: Attention layer (8 heads)
- Dropout: 0.3
- Dense layers: 2개 (16, 8)
- Output: Binary classification

# 학습 전략
- Learning rate: 0.001 → 0.0001 (decay)
- Batch size: 128
- Early stopping: val_loss 기준
- Gradient clipping: 1.0
```

### 3.3 트리 기반 보조 모델
```python
# LightGBM
- 전체 피처 활용
- 피처 중요도 추출용
- LSTM과 상관관계 낮은 오류 패턴

# CatBoost (Alternative)
- 카테고리 피처 자동 처리
- 작은 데이터셋에 강건
- Ordered boosting으로 과적합 방지

# 하이퍼파라미터 튜닝
- Optuna 20 trials
- Time-series CV
```

### 3.4 앙상블 전략
```python
# Soft voting
- LSTM/BiLSTM: 0.5
- LightGBM/CatBoost: 0.3
- Baseline: 0.2

# 동적 가중치
- 최근 N일 성능 기반 조정
- Model confidence 기반 가중
```

### 3.5 Walk-Forward 검증
```python
# 구조 (충분한 샘플 확보)
Train: 6개월 (≥20k samples)
Gap: 1주일  
Val: 1개월 (≥5k samples)
Gap: 1주일
Test: 1개월 (≥5k samples)

# 롤링 윈도우
- 6개월 학습 → 1개월 검증 → 1개월 테스트
- 5회 반복
- **각 윈도우마다 리스크-레버리지 재최적화**

# Expanding Window (비교 검증)
- 시작: 12개월
- 매월 누적 학습
- 재학습 주기 결정용

# Walk-Forward with 동적 최적화
```python
def walk_forward_validation(data, model_class):
    windows = create_walk_forward_windows(
        data,
        train_months=6,
        val_months=1,
        test_months=1,
        gap_days=7
    )
    
    results = []
    for window in windows:
        # 샘플 수 확인
        assert len(window.train_data) >= 20000, "Train 샘플 부족"
        assert len(window.val_data) >= 5000, "Val 샘플 부족"
        
        # 1. 모델 학습
        model = train_model(window.train_data, model_class)
        
        # 2. Validation에서 리스크-레버리지 최적화
        optimal_params = optimize_risk_leverage(
            window.val_data,
            model
        )
        
        # 3. Test에서 평가
        test_results = evaluate_model(
            window.test_data,
            model,
            optimal_params
        )
        
        # 4. 결과 저장
        results.append({
            'window_id': window.id,
            'optimal_params': optimal_params,
            'test_metrics': test_results
        })
        
        # 5. 파라미터 저장
        save_optimal_params(window.id, optimal_params)
    
    return results
```
```

---

## Phase 4: 거래 전략 최적화

### 4.1 리스크-레버리지 스위트스폿 탐색 (2단계 최적화)
```python
# Step 1: MC-Kelly로 Risk 구간 3개 선별
def select_risk_candidates(historical_data):
    # 초기 후보: Low, Mid, High
    risk_candidates = {
        'low': 0.5,    # 보수적
        'mid': 1.0,    # 표준
        'high': 1.5    # 공격적
    }
    
    # Monte Carlo로 각 Risk의 Kelly fraction 계산
    kelly_results = {}
    for risk_name, risk_pct in risk_candidates.items():
        kelly = monte_carlo_kelly(
            win_rate=0.52,  # 예상 승률
            risk_reward=1.5,  # R:R
            risk_pct=risk_pct,
            n_trades=250,    # 연간 거래수
            n_iter=10000
        )
        kelly_results[risk_pct] = kelly
    
    # Kelly > 0.2인 Risk만 선택 (최대 3개)
    selected_risks = [r for r, k in kelly_results.items() if k > 0.2]
    return sorted(selected_risks)[:3]

# Step 2: 선택된 Risk에서 레버리지 연속 탐색 (Bayesian)
from skopt import gp_minimize

def optimize_leverage_for_risk(risk_pct, historical_data):
    def objective(leverage):
        # 시뮬레이션 실행
        results = simulate_trading(
            historical_data,
            risk_pct,
            leverage[0]  # Bayesian opt는 리스트로 전달
        )
        # 목표: Sharpe 최대화 with 생존확률 제약
        if results['survival_rate'] < 0.9:
            return 10.0  # 페널티
        return -results['sharpe']  # 최소화 문제로 변환
    
    # Bayesian 최적화 (8-10회면 수렴)
    result = gp_minimize(
        func=objective,
        dimensions=[(5.0, 14.0)],  # 레버리지 범위
        n_calls=10,
        random_state=42
    )
    
    return result.x[0], -result.fun  # 최적 레버리지, Sharpe

# 전체 프로세스
def optimize_risk_leverage(historical_data):
    # Step 1: Risk 후보 선별
    risk_candidates = select_risk_candidates(historical_data)
    
    # Step 2: 각 Risk별 최적 레버리지 탐색
    results = {}
    for risk in risk_candidates:
        opt_lev, sharpe = optimize_leverage_for_risk(risk, historical_data)
        results[risk] = {
            'optimal_leverage': opt_lev,
            'sharpe': sharpe,
            'survival_rate': calculate_survival_rate(risk, opt_lev)
        }
    
    # 최종 선택 (Sharpe 최대)
    best_risk = max(results.items(), key=lambda x: x[1]['sharpe'])
    
    # 결과 저장
    with open('risk_profile.json', 'w') as f:
        json.dump({
            'optimal_risk_pct': best_risk[0],
            'optimal_leverage': best_risk[1]['optimal_leverage'],
            'metrics': best_risk[1]
        }, f)
    
    return best_risk
```

### 4.2 신호 생성
```python
# 확률 임계값
- Fixed: P > 0.52 (공격적)
- Quantile-based: 상위 X% (동적)
  * 일일 신호 수 목표 기반
  * 클래스 불균형 자동 조정
- Strong signal: P > 0.60 또는 상위 10%
- No trade zone: 0.48 < P < 0.52

# 더블 레짐 적용
```python
def get_entry_threshold(base_threshold, atr_regime, spread_regime):
    if spread_regime == 'high':
        # 스프레드 높으면 더 강한 신호만
        return base_threshold + 0.05
    elif atr_regime == 'high' and spread_regime == 'high':
        return None  # 진입 금지
    return base_threshold
```

# 필터
- 변동성 필터: ATR > threshold
- 시간 필터 (무료 대안)
  * 고정 경제지표 시간 회피
    - 08:30 ET (미국 고용/CPI)
    - 10:00 ET (미국 ISM/소비자신뢰)
    - 14:00 ET (FOMC 발표일)
  * 과거 데이터 분석으로 찾은 고변동 시간대
  * 장 시작/마감 30분
- 세션 필터: 주요 시장 시간
- 스프레드 필터: spread > 2*avg_spread 시 회피

# 더블 레짐 모델 (ATR × Spread)
- Normal ATR + Normal Spread: 정상 진입
- High ATR + Normal Spread: TP/SL 확대
- Normal ATR + High Spread: 신호 강도 상향
- High ATR + High Spread: 진입 금지

# Spread Regime 판정 기준
```python
def calculate_spread_regime(current_spread, window=50):
    """
    EWMA 기반 Spread Regime 판정
    - window: 50틱 (또는 5분)
    - High threshold: EWMA × 1.8
    """
    # EWMA 계산 (alpha=2/(window+1))
    ewma_spread = current_spread_series.ewm(
        span=window, 
        adjust=False
    ).mean()
    
    # 현재 스프레드가 EWMA의 1.8배 초과 시 High
    if current_spread > ewma_spread.iloc[-1] * 1.8:
        return 'high'
    else:
        return 'normal'

# 15초마다 업데이트, 진입 직전 재확인
def check_entry_conditions():
    # 진입 전 최종 체크 (Fail-Safe)
    current_spread = get_current_spread()
    spread_regime = calculate_spread_regime(current_spread)
    atr_regime = calculate_atr_regime()
    
    if spread_regime == 'high' and atr_regime == 'high':
        return False  # 진입 금지
    
    return True
```
```

### 4.3 포지션 사이징
```python
# 최적화된 파라미터 로드
with open('risk_profile.json', 'r') as f:
    optimal_params = json.load(f)

# 동적 Risk-Leverage 적용
base_risk_pct = optimal_params['optimal_risk_pct']
base_leverage = optimal_params['optimal_leverage']

# Kelly Criterion 변형
kelly_fraction_dynamic = base_kelly * (1 - DD_ratio)
position_size = account * base_risk_pct * min(
    kelly_fraction_dynamic,
    confidence_score,
    volatility_adjustment
)

# Volatility regime 조정
- Low vol: size × 1.5
- Normal: size × 1.0
- High vol: size × 0.7

# 실질 레버리지 제한
actual_leverage = position_size / (stop_distance * pip_value)
if actual_leverage > base_leverage:
    position_size = base_leverage * stop_distance * pip_value

# 제약사항
- 최소: 0.01 lot
- 최대: 계정의 20%
- 단계적 증가
```

### 4.4 리스크 관리
```python
# 계좌 수준 (동적 조정)
- Daily loss limit: 최적화된 risk% × 연속손실수
- Weekly loss limit: 시뮬레이션 기반 설정
- Max positions: 3-4
- Correlation check
- 포지션간 상관계수 < 0.7

# 동적 리스크 테이블
```python
def get_risk_parameters(vol_regime, dd_level, mc_results):
    base_risk = mc_results['optimal_risk_pct']
    base_lev = mc_results['optimal_leverage']
    
    # 시장 상태별 조정
    if vol_regime == 'high' or dd_level > 15:
        return base_risk * 0.6, base_lev * 0.5
    elif vol_regime == 'low' and dd_level < 5:
        return base_risk * 1.1, base_lev * 1.2
    else:
        return base_risk, base_lev
```

# 멀티포지션 관리
- 동일 방향 최대 2개
- 반대 방향 헤징 허용
- 진입 간격 최소 15분
- 전체 노출도: 동적 계산

# 포지션 수준
- Stop Loss: 0.7 * ATR (기본)
- Take Profit: 1.4 * ATR (2:1)

# 더블 레짐 조정
```python
def adjust_tp_sl(base_tp, base_sl, atr_regime, spread_regime):
    if atr_regime == 'high' and spread_regime == 'normal':
        return base_tp * 1.3, base_sl * 1.2
    elif atr_regime == 'normal' and spread_regime == 'high':
        return base_tp * 1.1, base_sl  # TP만 확대
    elif atr_regime == 'high' and spread_regime == 'high':
        return None, None  # 진입 금지
    else:
        return base_tp, base_sl
```

- Time stop: 2시간
- Trailing stop: 50% 이익 후
```

### 4.5 머니 매니지먼트
```python
# 복리 전략
- 수익 50% 출금
- 나머지 50% 재투자
- 원금 2배 도달 시 원금 회수

# 손실 복구 전략 (동적)
def adjust_position_by_dd(current_dd, optimal_params):
    if current_dd > optimal_params['max_dd'] * 0.8:
        return "MIN_LOT"
    elif current_dd > optimal_params['max_dd'] * 0.5:
        return "HALF_SIZE"
    else:
        return "NORMAL"

# RoR 기반 동적 조정
def adjust_risk_by_ror(account_balance, mc_simulation):
    ror = mc_simulation.get_risk_of_ruin()
    if ror > 0.4:
        return "STOP_TRADING"
    elif ror > 0.3:
        return "MIN_LOT_ONLY"
    elif ror > 0.2:
        return "HALF_POSITION"
    else:
        return "NORMAL"

# Walk-Forward 재최적화 트리거
- 생존확률 < 90%
- 3개월 연속 목표 미달성
- 시장 레짐 급변
```

---

## Phase 5: 백테스트 및 검증

### 5.1 백테스트 환경
```python
# 현실적 가정
- Spread: 실제 데이터 (tick)
- Slippage: 
  * Fixed: 0.5-1.0 pip
  * Random: Normal(μ=0.7, σ=0.2) pip
  * 더블 레짐 조정:
    - Normal: μ=0.7
    - High ATR: μ=1.2
    - High Spread: μ=1.5
    - Both High: μ=2.0
- Commission: $7/lot RT
- Margin call: 50%

# 롤오버 처리 (NAS100 CFD/선물)
```python
def apply_rollover_gaps(price_data, rollover_dates):
    """
    분기별 롤오버 gap 반영
    - 롤오버 날짜: 3월, 6월, 9월, 12월 셋째 금요일
    - Gap: 전일 종가 - 신규 시가
    """
    for date in rollover_dates:
        if date in price_data.index:
            # 전일 종가와 당일 시가의 gap
            prev_close = price_data.loc[:date].iloc[-2]['Close']
            curr_open = price_data.loc[date]['Open']
            gap = curr_open - prev_close
            
            # Gap 적용
            price_data.loc[date:, ['Open', 'High', 'Low', 'Close']] += gap
            
            # 롤오버 전후 3일간 스프레드 확대
            start_date = date - pd.Timedelta(days=3)
            end_date = date + pd.Timedelta(days=3)
            mask = (price_data.index >= start_date) & (price_data.index <= end_date)
            price_data.loc[mask, 'Spread'] *= 1.3
    
    return price_data
```

# 제약사항
- 고정 시간대 필터 (경제지표 시간)
- 주말 포지션 정리
- 갭 오픈 처리
- 시간대별 슬리피지 차등 적용
- 롤오버 기간 스프레드 확대
```

### 5.2 성과 지표
```python
# Primary Metrics
- Survival Rate: > 90%
- Sharpe Ratio: > 0.8
- Risk of Ruin: < 10%
- Win Rate: > 45%
- Profit Factor: > 1.3

# Secondary Metrics  
- Average Win/Loss ratio
- Consecutive losses (최대 허용치 동적)
- Recovery time
- Monthly consistency
- Risk of Ruin 분석
```

### 5.3 최적화 검증
```python
# 백테스트 결과로 최적화 파라미터 검증
optimal_params = json.load(open('risk_profile.json'))

# 실제 성과 vs 예상 성과 비교
actual_metrics = {
    'sharpe': backtest_results['sharpe'],
    'max_dd': backtest_results['max_dd'],
    'survival_rate': 1 - backtest_results['ruin_prob']
}

expected_metrics = optimal_params['metrics']

# 괴리도 분석
deviation = {
    k: abs(actual_metrics[k] - expected_metrics[k]) / expected_metrics[k]
    for k in actual_metrics
}

# 재최적화 필요 여부 판단
if any(dev > 0.2 for dev in deviation.values()):
    print("재최적화 필요: 실제와 예상 성과 괴리 20% 초과")
    # Phase 4.1 재실행
```

### 5.4 극단 시나리오 테스트
```python
# Stress Testing
- 연속 8패 시나리오
- Flash crash 시뮬레이션
- 스프레드 10배 확대
- 슬리피지 5 pip

# Monte Carlo (Wilson interval 기반 RoR)
- 10,000회 시뮬레이션
- Risk of Ruin 계산
  * 목표: < 10%
  * 경고: > 20%
  * 중단: > 30%
- 생존율 분석
- 최악 시나리오 대비

# RoR 계산 (Wilson interval)
```python
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

def calculate_risk_of_ruin(win_rate, risk_reward, risk_pct, n_trades):
    """
    Wilson interval 기반 RoR 계산
    입력: p(win), R:R, Risk%, 거래수
    """
    # Wilson interval로 승률 신뢰구간 계산
    n_wins = int(win_rate * n_trades)
    ci_low, ci_high = proportion_confint(n_wins, n_trades, 
                                        method='wilson',
                                        alpha=0.05)
    
    # 보수적 승률 사용 (하한)
    p_conservative = ci_low
    
    # Kelly criterion
    kelly = (p_conservative * risk_reward - (1 - p_conservative)) / risk_reward
    
    # Risk of Ruin 공식
    if kelly <= 0:
        return 1.0  # 100% 파산
    
    # 단순화된 RoR (무한 자본 가정)
    q = (1 - p_conservative) / p_conservative
    ror = q ** (1 / (risk_pct * kelly))
    
    return min(ror, 1.0)

def kelly_fraction_dynamic(base_kelly, mc_results):
    ruin_prob = mc_results['risk_of_ruin']
    dd_ratio = mc_results['current_dd'] / mc_results['max_dd']
    
    # RoR 기반 Kelly 조정
    if ruin_prob > 0.3:
        return 0  # 거래 중단
    elif ruin_prob > 0.2:
        kelly = base_kelly * 0.5
    elif ruin_prob > 0.1:
        kelly = base_kelly * 0.7
    else:
        kelly = base_kelly * (1 - dd_ratio)
    
    return max(kelly, 0.01)  # 최소값 보장
```
```

### 5.5 최적화 그리드 시각화
```python
# 2D 리스크-레버리지 히트맵 생성
risk_lev_heatmap = pd.DataFrame(
    index=risk_grid,
    columns=lev_grid
)

for r, l in itertools.product(risk_grid, lev_grid):
    survival, sharpe, cagr = results[(r, l)]
    # Pareto 최적해 찾기
    if survival > 0.9:
        risk_lev_heatmap.loc[r, l] = sharpe
    else:
        risk_lev_heatmap.loc[r, l] = np.nan

# 최적 영역 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(risk_lev_heatmap, 
            annot=True, 
            cmap='RdYlGn',
            center=0.8)
plt.title('Risk-Leverage Sweet Spot (Sharpe Ratio)')
plt.xlabel('Leverage')
plt.ylabel('Risk %')

# 동적 업데이트 저장
optimal_params = {
    'timestamp': datetime.now(),
    'market_regime': current_regime,
    'optimal_risk': best_risk,
    'optimal_leverage': best_lev,
    'expected_sharpe': best_sharpe,
    'survival_rate': best_survival
}
```

### 5.6 파라미터 범위
```python
# 2단계 최적화 파라미터
# Step 1: MC-Kelly Risk 선별
- Risk % 후보: [0.5, 1.0, 1.5] (Low, Mid, High)
- MC 시뮬레이션: 10,000회
- Kelly > 0.2 필터

# Step 2: Bayesian 레버리지 탐색
- Leverage 범위: [5.0, 14.0] (연속)
- Bayesian calls: 8-10회
- 수렴 기준: Sharpe 개선 < 1%

# 기타 파라미터
- ATR multiplier (SL): [0.5, 0.7, 1.0]
- ATR multiplier (TP): [1.5, 2.0, 2.5]
- Time stop: [1h, 2h, 3h]

# 최적화 제약조건
- 생존확률 > 90% (Wilson interval 기반)
- Sharpe Ratio > 0.8
- Max DD < 40%
- Risk of Ruin < 10%

# 목적함수 (다목적 최적화)
- Primary: 생존확률 최대화
- Secondary: Sharpe Ratio 최대화
- Constraint: RoR < 10% (hard constraint)

# 평균 실행 시간
- 전체 최적화: 30-45분 (기존 3-4시간 → 대폭 단축)
- Walk-Forward 1회: 6-8분
```

---

## Phase 6: 실전 전환

### 6.1 시스템 아키텍처
```python
# 실시간 파이프라인
1. Data Collection (1분 주기)
   - Price feed
   - Spread monitoring
   - Latency check (>5s alert)
   
2. Risk Parameter Loading
   - risk_profile.json 읽기
   - 현재 시장 상태 확인
   - 동적 파라미터 적용
   
3. Feature Pipeline  
   - 코어 피처 계산
   - 피처 품질 체크
   - Staleness detection
   
4. Prediction
   - 모델 앙상블
   - Confidence 계산
   - 더블 레짐 체크
   
5. Execution
   - Signal validation
   - Order management
   - TP/SL 동적 조정
   
6. Monitoring
   - Performance tracking
   - Anomaly detection
   - Data freshness alerts
   - 생존확률 실시간 계산
   
7. Alert System
   - 생존확률 < 85% 경고
   - 연속 손실 경고
   - 시스템 오류 즉시 알림
   - 포지션 청산 알림
```

### 6.2 단계별 실전 전환
```python
# 초기 자금: $300 (전액 손실 가능 금액)
# 목표: 지속가능한 성장 (생존확률 > 90%)

# Week 1-2: Paper Trading
- Full system test
- Latency measurement
- Bug fixes
- 최대 손실 시나리오 테스트

# Week 3-4: Micro lots (0.01)
- Real money psychology
- Slippage analysis
- 연속 손실 대응 훈련

# Week 5-8: Mini lots (0.1)  
- Scale test
- Risk metrics validation
- 회복 전략 검증

# Week 9+: Standard lots
- Gradual increase
- Performance monitoring
- 심리적 압박 관리
```

---

## 📊 예상 성과 지표

### 모델 성능
```python
# 예측 정확도
- Training: 54-56%
- Validation: 52-54%
- Test: 52-53%
- Gap < 2%

# 신호 품질
- Daily signals: 최적화 결과에 따라 변동
- High confidence (>0.6): 상위 10-20%
- Average holding: 45-90분
```

### 거래 성과
```python
# Returns (시뮬레이션 기반)
- Monthly: 최적화 결과에 따름
- Annual: 생존확률 90% 제약 하 최대값
- Sharpe: 0.8-1.5 (타겟)

# Risk (동적 계산)
- Max DD: 시뮬레이션 95% 신뢰구간
- Daily VaR: 최적 risk% 기반
- Loss months: 30-40%

# 거래 프로필 (최적화 결과 기반)
- 일 평균: 시뮬레이션 결과
- 평균 보유: 45-90분
- 승률: 45-52%
- 손익비: 최적화된 R:R
```

---

## 🚨 리스크 및 중단 신호

### Phase Gates
```python
# Phase 1-2
□ Triple Barrier 구현 완료
□ Volatility regime별 임계값 검증
□ 피처 20-25개로 축소
□ PCA 95% variance 확인
□ 클래스 균형 달성
□ 고위험 라벨 비율 확인
□ 데이터 누수 테스트 유닛 구현

# Phase 3-4  
□ Val accuracy > 52%
□ Overfit gap < 3%
□ Feature importance 분석
□ Attention weights 해석 (BiLSTM)
□ 리스크-레버리지 최적화 완료
□ 생존확률 > 90% 확인
□ Quantile threshold 최적화
□ 피처 생성 누수 검증 통과

# Phase 5-6
□ Risk-Leverage 최적화 완료
□ 생존확률 > 90% 달성
□ Backtest Sharpe > 0.8
□ Paper-Real gap < 15%
□ System stability 72h
□ Data latency < 5s 유지율 99%
□ Risk-of-Ruin < 10%
□ Kelly dynamic 조정 검증
□ 동적 파라미터 실시간 적용
```

### Red Flags
1. **Accuracy < 51%**: 라벨/피처 재검토
2. **피처 > 30개**: 과적합 위험
3. **DD > 최적화된 한계**: 리스크 재설계
4. **신호 < 5/day**: 임계값 조정

### 비상 정지 조건
- Risk of Ruin > 30% 시 거래 중단
- 생존확률 < 80% 시 전략 재검토
- 연속 손실 > 시뮬레이션 99% 신뢰구간
- 일일 손실 > 최적화된 한계
- 시스템 오류 3회 이상 시 점검

### 계좌 보호
- 일일 최대 손실: 최적화된 값
- 최대 동시 포지션: 3-4개
- 연속 손실 한계: 시뮬레이션 99% 신뢰구간

### 포지션별
- Stop Loss: 0.5-1.0 * ATR (최적화)
- Take Profit: 1.5-2.5 * ATR (최적화)
- Time Stop: 1-3시간 (최적화)

---

## 🔧 도구 및 환경

### 개발 환경
- Python 3.10+
- TensorFlow/PyTorch
- Pandas/NumPy
- Optuna
- Backtrader/Vectorbt

### 데이터 관리
- PostgreSQL/TimescaleDB
- Redis (실시간 캐시)
- S3 (백업)

### 모니터링
- Grafana dashboards
- Slack alerts
- Performance logs

---

## 💀 동적 리스크 전략 핵심 요약

**"Survive to Thrive"**
- 고정된 리스크가 아닌 시장 적응형 최적화
- 생존확률 90% 이상 유지하며 최대 수익 추구
- Walk-Forward마다 리스크-레버리지 재계산
- 시장 상태에 따른 자동 스케일링

**핵심 보호 장치**
1. 2단계 리스크-레버리지 최적화 (MC-Kelly → Bayesian)
2. Wilson interval 기반 RoR 계산 (소표본 편향 완화)
3. 더블 레짐 모델 (EWMA spread 50틱 × 1.8 = High)
4. 데이터 누수 자동 검증 시스템

**동적 조정 규칙**
1. 매 Walk-Forward 후 최적값 재계산 (8-10회 수렴)
2. 생존확률 < 90% 시 즉시 재최적화
3. 15초마다 Spread Regime 체크 (50틱 EWMA 기준)
4. DD 수준별 단계적 리스크 축소

**실제 운영 예시**
- 저변동 + Low DD: Risk 1.2%, Lev 10×
- 정상 시장: Risk 1.0%, Lev 8×
- 고변동 or High DD: Risk 0.6%, Lev 5×
- 롤오버 기간: Spread × 1.3 적용

이 전략은 **생존이 최우선**입니다.
파산하지 않고 꾸준히 성장하는 것이 핵심입니다.

---

## ⚠️ 최종 경고

**이 시스템을 사용하기 전에 반드시 이해하세요:**
1. 전체 투자금을 잃을 수 있습니다
2. 레버리지는 손실을 크게 증폭시킵니다
3. 백테스트 결과는 실제 거래와 다를 수 있습니다
4. 심리적 압박을 견딜 준비가 필요합니다
5. 충분한 테스트 없이 실거래하지 마세요

**투자는 개인의 책임입니다. 이 시스템은 투자 조언이 아닙니다.**

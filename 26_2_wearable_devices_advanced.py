# -*- coding: utf-8 -*-
"""
26_2_wearable_devices_advanced.py
웨어러블 디바이스 프로젝트: 호흡 & HRV 중심의 고도화된 백색소음 집중도 분류 파이프라인
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# ==========================================
# [설정] 데이터 폴더 및 옵션
# ==========================================
# 구글 코랩 환경 대응 경로 체크
if not os.path.exists("HiCardiEducationSave_230329"):
    if os.path.exists("/content/drive/MyDrive/HiCardiEducationSave_230329"):
        DATA_DIR = "/content/drive/MyDrive/HiCardiEducationSave_230329"
    else:
        DATA_DIR = "HiCardiEducationSave_230329"
else:
    DATA_DIR = "HiCardiEducationSave_230329"
EXCLUDE_SESSION_5 = True  # 생리학적 아웃라이어로 확인된 5회차 데이터 제외 여부

# 1. 시나리오 데이터 리스트 정의
scenarios = [
    # 1회차
    {"base": "260603_공연1-1.txt", "task": "260603_백색소음1.txt", "label": 0, "session": "S1_1", "train": True},
    {"base": "260604_공연1-2.txt", "task": "260604_빠른음악1.txt", "label": 1, "session": "S1_2", "train": True},
    # 2회차
    {"base": "260607_공연2-1.txt", "task": "260607_백색소음2.txt", "label": 0, "session": "S2_1", "train": True},
    {"base": "260608_공연2-2.txt", "task": "260608_빠른음악2.txt", "label": 1, "session": "S2_2", "train": True},
    # 3회차
    {"base": "260609_공연3-1.txt", "task": "260609_백색소음3.txt", "label": 0, "session": "S3_1", "train": True},
    {"base": "260610_공연3-2.txt", "task": "260610_빠른음악3.txt", "label": 1, "session": "S3_2", "train": True},
    # 4회차
    {"base": "260611_공연4-1.txt", "task": "260611_백색소음4.txt", "label": 0, "session": "S4_1", "train": True},
    {"base": "260613_공연4-2.txt", "task": "260613_빠른음악4.txt", "label": 1, "session": "S4_2", "train": True},
    # 5회차 (선택적 제외)
    {"base": "260614_공연5-1.txt", "task": "260614_백색소음5.txt", "label": 0, "session": "S5_1", "train": False, "is_s5": True},
    {"base": "260615_공연5-2.txt", "task": "260615_빠른음악5.txt", "label": 1, "session": "S5_2", "train": False, "is_s5": True},
    # 6회차
    {"base": "260616_공연6-1.txt", "task": "260616_백색소음6.txt", "label": 0, "session": "S6_1", "train": False},
    {"base": "260617_공연6-2.txt", "task": "260617_빠른음악6.txt", "label": 1, "session": "S6_2", "train": False},
    # 7회차
    {"base": "260618_공연7-1.txt", "task": "260618_백색소음7.txt", "label": 0, "session": "S7_1", "train": False},
    {"base": "260619_공연7-2.txt", "task": "260619_빠른음악7.txt", "label": 1, "session": "S7_2", "train": False},
]

# 설정에 따라 train_scenarios와 test_scenarios 동적 분류
train_scenarios = []
test_scenarios = []

for sc in scenarios:
    if sc.get("is_s5", False) and EXCLUDE_SESSION_5:
        continue  # 5회차 완전 배제 설정 시 스킵
    
    if sc["train"]:
        train_scenarios.append(sc)
    else:
        test_scenarios.append(sc)

# ==========================================
# 2. 고도화된 피처 추출 함수 (Cleansing 포함)
# ==========================================
def extract_features(filepath):
    try: 
        df = pd.read_csv(filepath, sep='\t', header=None)
    except: 
        return None

    # 데이터 정제 (Data Cleansing)
    # 1. 패치 이탈(Lead off == 1) 구간 제거
    df = df[df[8] == 0].reset_index(drop=True)
    # 2. 격한 신체 움직임(Motion Status >= 2) 제거 -> 집중 시 정적 상태만 유지
    df = df[df[6] <= 1].reset_index(drop=True)

    if len(df) < 150:
        return None

    # 노이즈 필터링 (Rolling Median)
    df[1] = df[1].rolling(window=5, center=True, min_periods=1).median()
    df[2] = df[2].rolling(window=5, center=True, min_periods=1).median()
    df[4] = df[4].rolling(window=5, center=True, min_periods=1).median()

    window_rows, step_rows = 150, 100
    n_windows = (len(df) - window_rows) // step_rows + 1
    if n_windows <= 0: 
        return None

    windows_data = []
    for i in range(n_windows):
        start = i * step_rows
        w = df.iloc[start:start + window_rows]
        hr, rr = w[1].values, w[2].values
        resp = w[4].values

        # 중복비트 제거 후 순수 RR 간격 리스트 생성
        true_beats = [rr[0]]
        for j in range(1, len(rr)):
            if rr[j] != rr[j-1]: 
                true_beats.append(rr[j])
        true_beats = np.array(true_beats)

        if len(true_beats) < 5: 
            continue
        
        diff_rr = np.diff(true_beats)
        
        # ECG Raw 전압 및 가속도 데이터 범위
        raw = w.iloc[:, 11:75].values.flatten()
        raw = raw[~np.isnan(raw)]

        # --- HRV 피처 계산 ---
        mean_hr = np.mean(hr)
        mean_rr = np.mean(true_beats)
        sdnn = np.std(true_beats, ddof=1) if len(true_beats) > 1 else 0
        rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0
        
        # pNN50
        nn50 = np.sum(np.abs(diff_rr) > 50) if len(diff_rr) > 0 else 0
        pnn50 = nn50 / len(diff_rr) if len(diff_rr) > 0 else 0
        
        # Range RR
        range_rr = np.ptp(true_beats) if len(true_beats) > 0 else 0
        
        # CV RR (변동계수)
        cv_rr = sdnn / mean_rr if mean_rr > 0 else 0
        
        # Baevsky's Stress Index (스트레스 지수)
        rounded_beats = np.round(true_beats / 50.0) * 50.0
        vals, counts = np.unique(rounded_beats, return_counts=True)
        mode_idx = np.argmax(counts)
        Mo = vals[mode_idx] / 1000.0  # 초 단위 변환
        AMo = (counts[mode_idx] / len(rounded_beats)) * 100.0  # 백분율 변환
        MxDMn = (np.max(true_beats) - np.min(true_beats)) / 1000.0  # 초 단위 변환
        stress_index = AMo / (2.0 * Mo * MxDMn) if (Mo > 0 and MxDMn > 0) else 0

        # --- 호흡 피처 계산 ---
        mean_resp = np.mean(resp)
        std_resp = np.std(resp) if len(resp) > 0 else 0
        range_resp = np.ptp(resp) if len(resp) > 0 else 0

        # --- ECG 신호 특성 ---
        raw_std = np.std(raw) if len(raw) > 0 else 0
        raw_energy = np.sum(raw**2) / len(raw) if len(raw) > 0 else 0

        windows_data.append({
            'Mean_HR': mean_hr,
            'Mean_RR': mean_rr,
            'SDNN': sdnn,
            'RMSSD': rmssd,
            'pNN50': pnn50,
            'Range_RR': range_rr,
            'CV_RR': cv_rr,
            'Stress_Index': stress_index,
            'Mean_Resp': mean_resp,
            'Std_Resp': std_resp,
            'Range_Resp': range_resp,
            'Raw_Std': raw_std,
            'Raw_Energy': raw_energy
        })
        
    return pd.DataFrame(windows_data)

# ==========================================
# 3. 1:1 베이스라인 차감(Delta) 정규화
# ==========================================
feature_cols = [
    'Mean_HR', 'Mean_RR', 'SDNN', 'RMSSD', 'pNN50', 'Range_RR', 'CV_RR', 
    'Stress_Index', 'Mean_Resp', 'Std_Resp', 'Range_Resp', 'Raw_Std', 'Raw_Energy'
]

def build_delta_matrix(scenarios):
    matrix_list = []

    for sc in scenarios:
        base_path = os.path.join(DATA_DIR, sc["base"])
        task_path = os.path.join(DATA_DIR, sc["task"])

        if not os.path.exists(base_path) or not os.path.exists(task_path):
            print(f"[경고] 파일 누락: {base_path} 또는 {task_path}")
            continue

        df_base = extract_features(base_path)
        df_task = extract_features(task_path)

        if df_base is not None and df_task is not None and not df_base.empty and not df_task.empty:
            base_means = df_base[feature_cols].mean()
            df_delta = df_task[feature_cols].copy()

            for col in feature_cols:
                df_delta[col] = df_delta[col] - base_means[col]

            df_delta['Label'] = sc["label"]
            matrix_list.append(df_delta)

    return pd.concat(matrix_list, ignore_index=True) if matrix_list else pd.DataFrame()

# ==========================================
# 4. 파이프라인 가동 및 모델링 비교
# ==========================================
print("--- [시작] 데이터 로드 및 피처 추출 가동 ---")
train_df = build_delta_matrix(train_scenarios)
test_df  = build_delta_matrix(test_scenarios)

if not train_df.empty and not test_df.empty:
    X_train = train_df.drop(['Label'], axis=1)
    y_train = train_df['Label']
    X_test  = test_df.drop(['Label'], axis=1)
    y_test  = test_df['Label']

    print(f"\n[데이터셋 구축 완료]")
    print(f"  - Train 샘플 수: {len(X_train)}행 (피처 수: {X_train.shape[1]})")
    print(f"  - Test 샘플 수: {len(X_test)}행")

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4가지 모델 학습 및 검증
    models = {
        "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42)
    }

    print("\n--- [평가] 머신러닝 모델별 예측 성능 비교 ---")
    best_acc = 0.0
    best_model_name = ""
    preds_dict = {}

    for name, model in models.items():
        if "SVM" in name or "Logistic" in name:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        preds_dict[name] = preds
        print(f"  - {name:<20} 정확도: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    print(f"\n[Best Model] 최고 성능 기본 모델: {best_model_name} (정확도: {best_acc:.4f})")

    # 5. 로지스틱 회귀 하이퍼파라미터 튜닝 단계 생략

    # ==========================================
    # 6. 최종 성적표 출력
    # ==========================================
    print(f"\n==============================================")
    print(f"[최종 성적표] 재설계된 13개 고도화 피처 기반")
    print(f"==============================================")
    
    # 로지스틱 회귀 레포트
    print(f"\n[1] Logistic Regression 정확도: {accuracy_score(y_test, preds_dict['Logistic Regression']):.4f}")
    print(classification_report(y_test, preds_dict['Logistic Regression'], target_names=['백색소음(0)', '빠른음악(1)']))

    # 랜덤 포레스트 레포트
    print(f"----------------------------------------------")
    print(f"[2] Random Forest 정확도: {accuracy_score(y_test, preds_dict['Random Forest']):.4f}")
    print(classification_report(y_test, preds_dict['Random Forest'], target_names=['백색소음(0)', '빠른음악(1)']))

    # SVM 레포트
    print(f"----------------------------------------------")
    print(f"[3] SVM (RBF Kernel) 정확도: {accuracy_score(y_test, preds_dict['SVM (RBF Kernel)']):.4f}")
    print(classification_report(y_test, preds_dict['SVM (RBF Kernel)'], target_names=['백색소음(0)', '빠른음악(1)']))

    # ==========================================
    # 7. 피처 중요도 분석 (Random Forest 기준)
    # ==========================================
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"==============================================")
    print(f"[피처 기여도 분석] Random Forest 기준 상위 피처 순위")
    print(f"==============================================")
    for rank in range(X_train.shape[1]):
        col_name = X_train.columns[indices[rank]]
        val = importances[indices[rank]]
        print(f"  {rank+1:>2}위. {col_name:<15} (기여도: {val:.4f})")
        
    # ==========================================
    # 8. 시각화 (Visualization) 추가
    # ==========================================
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        # 한글 깨짐 방지 설정 (Windows용 맑은 고딕 및 코랩 대응)
        import platform
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:
            try:
                # 코랩 나눔 폰트 대응 시도
                plt.rc('font', family='NanumBarunGothic') 
            except:
                plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False

        # (1) 모델별 정확도 비교 시각화
        plt.figure(figsize=(8, 5))
        models_list = list(models.keys())
        accuracies = [accuracy_score(y_test, preds_dict[name]) for name in models.keys()]
        
        sns.barplot(x=models_list, y=accuracies, palette='Set2')
        plt.title('모델별 이진 분류 정확도 비교 (백색소음 vs 빠른음악)', fontsize=13, pad=15)
        plt.ylabel('정확도 (Accuracy)', fontsize=11)
        plt.ylim(0, 1.1)
        for idx, val in enumerate(accuracies):
            plt.text(idx, val + 0.02, f"{val:.4f}", ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig('model_accuracy_comparison.png', dpi=300)
        plt.close()
        print("[시각화 완료] model_accuracy_comparison.png 저장됨")

        # (2) 피처 중요도 시각화 (Top 10)
        plt.figure(figsize=(9, 5.5))
        top_n = 10
        top_indices = indices[:top_n]
        top_features = X_train.columns[top_indices]
        top_importances = importances[top_indices]
        
        # 영어 피처 명칭을 한글로 매핑
        bio_desc_map = {
            'Mean_HR': '평균 심박수', 'Mean_RR': '평균 RR간격', 'SDNN': 'SDNN(조절능)', 
            'RMSSD': 'RMSSD(부교감활성)', 'pNN50': 'pNN50', 'Range_RR': 'RR 범위', 
            'CV_RR': 'RR 변동계수', 'Stress_Index': '스트레스 지수', 'Mean_Resp': '평균 호흡', 
            'Std_Resp': '호흡 변동성(std)', 'Range_Resp': '호흡 범위', 'Raw_Std': 'ECG 표준편차', 
            'Raw_Energy': 'ECG 에너지'
        }
        korean_features = [bio_desc_map.get(col, col) for col in top_features]
        
        sns.barplot(x=top_importances, y=korean_features, palette='viridis')
        plt.title('Random Forest 상위 10개 피처 중요도', fontsize=13, pad=15)
        plt.xlabel('중요도 (Gini Importance)', fontsize=11)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()
        print("[시각화 완료] feature_importance.png 저장됨")

        # (3) 로지스틱 회귀 혼동 행렬(Confusion Matrix) 시각화
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_test, preds_dict["Logistic Regression"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['백색소음(0)', '빠른음악(1)'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Logistic Regression 혼동 행렬', fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.close()
        print("[시각화 완료] confusion_matrix.png 저장됨")

        # (4) 회차별 기저(Base)를 차감한 순수 심박수 변화량(Delta HR) 대조 바 차트 생성
        session_data = []
        for sc in scenarios:
            if sc.get("is_s5", False) and EXCLUDE_SESSION_5:
                continue
            base_p = os.path.join(DATA_DIR, sc["base"])
            task_p = os.path.join(DATA_DIR, sc["task"])
            if os.path.exists(base_p) and os.path.exists(task_p):
                df_b = extract_features(base_p)
                df_t = extract_features(task_p)
                if df_b is not None and df_t is not None and not df_b.empty and not df_t.empty:
                    session_group = sc["session"][:2]
                    session_data.append({
                        "Session": session_group,
                        "Condition": "백색소음 (Label 0)" if sc["label"] == 0 else "빠른음악 (Label 1)",
                        "Label": sc["label"],
                        "Delta_HR": df_t['Mean_HR'].mean() - df_b['Mean_HR'].mean()
                    })
        session_df = pd.DataFrame(session_data)

        plt.figure(figsize=(9, 5))
        sns.barplot(x='Session', y='Delta_HR', hue='Condition', data=session_df, 
                    palette={'백색소음 (Label 0)': '#3498db', '빠른음악 (Label 1)': '#e74c3c'})
        plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)
        plt.title('회차별 기저(공연) 대비 심박수 변화량(Delta HR) 대조 분석', fontsize=12, pad=12)
        plt.ylabel('평균 심박수 변화량 (Delta BPM)', fontsize=10)
        plt.xlabel('실험 회차 (Session)', fontsize=10)
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('physiological_feature_comparison.png', dpi=300)
        plt.close()
        print("[시각화 완료] physiological_feature_comparison.png 저장됨")

        # (5) 5회차 배제 분석용 전체 7개 세션 Delta_HR 비교 차트 생성
        all_session_data = []
        for sc in scenarios:
            base_p = os.path.join(DATA_DIR, sc["base"])
            task_p = os.path.join(DATA_DIR, sc["task"])
            if os.path.exists(base_p) and os.path.exists(task_p):
                df_b = extract_features(base_p)
                df_t = extract_features(task_p)
                if df_b is not None and df_t is not None and not df_b.empty and not df_t.empty:
                    session_group = sc["session"][:2]
                    all_session_data.append({
                        "Session": session_group,
                        "Condition": "백색소음 (Label 0)" if sc["label"] == 0 else "빠른음악 (Label 1)",
                        "Label": sc["label"],
                        "Delta_HR": df_t['Mean_HR'].mean() - df_b['Mean_HR'].mean()
                    })
        all_session_df = pd.DataFrame(all_session_data)

        plt.figure(figsize=(9, 5))
        sns.barplot(x='Session', y='Delta_HR', hue='Condition', data=all_session_df, 
                    palette={'백색소음 (Label 0)': '#3498db', '빠른음악 (Label 1)': '#e74c3c'})
        plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)
        plt.axvspan(3.5, 4.5, color='yellow', alpha=0.2, label='생리학적 아웃라이어 (5회차 - 배제됨)')
        plt.title('전체 회차(S1~S7) 심박수 변화량(Delta HR) 비교 및 5회차 역전 현상 분석', fontsize=12, pad=12)
        plt.ylabel('평균 심박수 변화량 (Delta BPM)', fontsize=10)
        plt.xlabel('실험 회차 (Session)', fontsize=10)
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('session_exclusion_analysis.png', dpi=300)
        plt.close()
        print("[시각화 완료] session_exclusion_analysis.png 저장됨")

    except Exception as vis_err:
        print("[시각화 에러] 시각화를 수행할 수 없습니다:", vis_err)
    
else:
    print("[오류] 파일 로드에 실패했습니다. 폴더 내 파일명과 코드를 다시 한번 확인해주세요.")

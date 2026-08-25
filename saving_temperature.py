"""
적금의 온도 (Saving Temperature)
26주적금 중도해지 선제 방어 시스템 — 예측 → 원인 분해 → Uplift 타겟팅 → 정책 엔진

실행: python saving_temperature.py
의존성: numpy, pandas, scikit-learn, matplotlib
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

RANDOM_SEED = 42

# ============================================================
# 1. 합성 데이터 생성 (개인정보 없이 방법론 증명)
# ============================================================
def generate_data(n=10000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    week = rng.integers(1, 27, n)
    base = rng.choice([1000, 2000, 3000, 5000, 10000], n, p=[.40, .25, .20, .10, .05])
    weekly_amount = base * week                       # 26주적금: 주차마다 납입액 증가
    monthly_income = rng.normal(280, 90, n).clip(80, 900)
    burden = (weekly_amount * 4.3 / 10000) / monthly_income
    bn = (burden - burden.min()) / (burden.max() - burden.min())

    balance_buffer   = rng.gamma(2.0, 40, n).clip(1, 500)
    large_expense    = rng.binomial(1, .18, n)
    income_regular   = rng.beta(5, 2, n)
    login_drop       = rng.beta(2, 4, n)
    payment_delay    = rng.beta(1.5, 6, n)
    savings_page_view= rng.poisson(.7, n)
    auto_transfer    = rng.binomial(1, .62, n)
    competitor_gap   = rng.normal(.3, .6, n).clip(-1, 2)

    # 해지 원인 3분해
    z_liq = 1.6*large_expense + 1.3*(balance_buffer < 30) + 1.1*(1-income_regular)
    z_mot = 3.2*bn + 1.2*(week/26) + 1.3*login_drop + 1.0*payment_delay - 0.8*auto_transfer
    z_swi = 1.4*competitor_gap + 0.45*savings_page_view
    cs = np.vstack([z_liq, z_mot, z_swi]).T
    cause_names = np.array(['유동성압박', '동기상실', '상품이동'])

    # 해지 라벨 (여러 원인 누적 + 잡음)
    logit = cs.sum(1) - 4.0 + rng.normal(0, .4, n)
    p_control = 1/(1+np.exp(-logit))
    churn = rng.binomial(1, p_control)
    primary_cause = np.where(churn == 1, cause_names[cs.argmax(1)], '해지안함')

    # Uplift 세그먼트 (개입 효과가 다른 4종)
    rank = pd.Series(p_control).rank(pct=True).values
    seg = np.empty(n, dtype=object); u = rng.random(n)
    at_risk = rank >= .60
    seg[at_risk]  = np.where(u[at_risk]  < .55, '설득가능', '결심형')
    seg[~at_risk] = np.where(u[~at_risk] < .15, '잠자는개', '확실유지')
    effect = {'설득가능': -.35, '결심형': -.02, '잠자는개': +.15, '확실유지': 0.}
    p_treat = np.clip(p_control + np.array([effect[s] for s in seg]), .01, .99)

    return pd.DataFrame(dict(
        week=week, weekly_amount=weekly_amount, burden=bn.round(3),
        balance_buffer=balance_buffer.round(1), large_expense=large_expense,
        income_regular=income_regular.round(3), login_drop=login_drop.round(3),
        payment_delay=payment_delay.round(3), savings_page_view=savings_page_view,
        auto_transfer=auto_transfer, competitor_gap=competitor_gap.round(3),
        p_churn_control=p_control.round(3), p_churn_treat=p_treat.round(3),
        uplift_segment=seg, primary_cause=primary_cause, churn=churn))

FEATURES = ['week','weekly_amount','burden','balance_buffer','large_expense',
            'income_regular','login_drop','payment_delay','savings_page_view',
            'auto_transfer','competitor_gap']

# ============================================================
# 2. 해지 예측 모델
# ============================================================
def train_model(df):
    X, y = df[FEATURES], df['churn']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=RANDOM_SEED, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=RANDOM_SEED).fit(Xtr, ytr)
    sc = StandardScaler(); lr = LogisticRegression(max_iter=1000).fit(sc.fit_transform(Xtr), ytr)
    print(f"[모델] RandomForest AUC = {roc_auc_score(yte, rf.predict_proba(Xte)[:,1]):.3f}")
    print(f"[모델] Logistic     AUC = {roc_auc_score(yte, lr.predict_proba(sc.transform(Xte))[:,1]):.3f}")
    df['churn_score'] = rf.predict_proba(df[FEATURES])[:, 1]
    return rf

# ============================================================
# 3. Uplift 타겟팅 (위험도 vs 개입효과)
# ============================================================
def uplift_analysis(df, budget=.20):
    df['true_uplift'] = df.p_churn_control - df.p_churn_treat
    n = len(df); k = int(n*budget)
    up = np.cumsum(df.true_uplift.values[np.argsort(-df.true_uplift.values)])[k-1]
    rk = np.cumsum(df.true_uplift.values[np.argsort(-df.p_churn_control.values)])[k-1]
    print(f"[Uplift] 상위 {int(budget*100)}% 개입 시 예방 해지: Uplift={up:.0f}명 vs 위험도={rk:.0f}명 (+{(up/rk-1)*100:.0f}%)")
    return up, rk

# ============================================================
# 4. 원인별 정책 엔진 (Trigger & Action)
# ============================================================
ACTIONS = {
    '유동성압박': ('부분출금·긴급출금 안내', '필요한 만큼만 꺼내고 저축은 이어가요'),
    '동기상실':   ('쉬어가기 + 토닥 메시지', '벌써 {week}주나 달려오셨는걸요!'),
    '상품이동':   ('우대금리·저금통 연계 제안', '더 나은 조건, 카뱅에서 찾았어요'),
}
def infer_cause(r):
    s = {'유동성압박': r.large_expense*1.6 + (r.balance_buffer<30)*1.3 + (1-r.income_regular)*1.1,
         '동기상실':   r.burden*3.2 + (r.week/26)*1.2 + r.login_drop*1.3 + r.payment_delay*1.0 - r.auto_transfer*0.8,
         '상품이동':   r.competitor_gap*1.4 + r.savings_page_view*0.45}
    return max(s, key=s.get)
def policy_engine(df):
    TH = df.churn_score.quantile(.80)
    def policy(r):
        if r.churn_score < TH: return pd.Series(['개입 안 함','-','-'])
        cause = infer_cause(r); act, msg = ACTIONS[cause]
        return pd.Series([cause, act, msg.format(week=r.week)])
    df[['추정원인','개입액션','메시지']] = df.apply(policy, axis=1)
    tgt = df[df.추정원인 != '개입 안 함']
    print(f"[정책] 개입 대상 {len(tgt)}명 / 원인별: {tgt.추정원인.value_counts().to_dict()}")
    return df

if __name__ == '__main__':
    df = generate_data()
    print(f"[데이터] {len(df)}명 생성, 전체 해지율 {df.churn.mean():.1%}")
    train_model(df)
    uplift_analysis(df)
    policy_engine(df)
    print("\n완료 ✅")

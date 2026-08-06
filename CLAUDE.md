# okdalto.github.io — 에이전트 매뉴얼

Jekyll 기반 한/영 이중 언어 블로그. 아래 규칙은 기존 글 56쌍 전수 분석(2026-08)으로
도출한 실제 관례다. 새 글을 쓸 때 이 매뉴얼을 그대로 따른다.

## 1. 파일과 프론트매터

포스트는 **한국어(`_posts/`) + 영어(`_en/`)** 쌍으로 만든다. 언어 전환 버튼은
프론트매터의 `ref`로 짝을 찾으므로(`_layouts/default.html`), `ref`가 어긋나면
버튼이 조용히 사라진다. **`ref`가 가장 중요한 필드다.**

1. 한국어 원문: `_posts/YYYY-MM-DD-<한글 제목>.md` — 파일명에 공백 그대로 사용
   (하이픈 치환 안 함). 날짜 프리픽스는 게시일.
2. 영어 번역: `_en/<ref>.md` — 파일명 = `ref` 값. 날짜 프리픽스 없음.

### 프론트매터 — 필드 6개, 이 순서 고정 (전 포스트 100% 동일)

한국어 (`_posts/`):

```yaml
---
title: "학습과 변하지 않는 것에 대해서"
permalink: /thoughts/learning-and-what-does-not-change/
date: 2026-08-06T00:00:00+09:00
categories:
  - 생각
tags:
  - AI
  - 학습
ref: learning-and-what-does-not-change
---
```

영어 (`_en/`):

```yaml
---
title: "On Learning and What Does Not Change"
permalink: /en/learning-and-what-does-not-change/
date: 2026-08-06T00:00:00+09:00
categories:
  - thoughts
tags:
  - AI
  - learning
ref: learning-and-what-does-not-change
---
```

- `layout`, `excerpt` 등 다른 필드는 쓰지 않는다(`_config.yml` defaults가 처리).
- `title`은 항상 큰따옴표. 제목 안 인용은 곧은 홑따옴표(`'촘촘히 감는다'가 0점이 되는 교육`),
  부제는 em-dash(`이미지 생성 AI의 역사 — '아직 멀었어'의 연대기`).
- `date`는 항상 `T…+09:00`. 신작 에세이는 `T00:00:00`. 한/영 쌍의 `date`는 **바이트 단위로 동일**.
  미래 날짜는 GitHub Pages가 게시하지 않으므로 금지.
- (참고) 옮겨온 글은 `date`=원문 작성 시점, 파일명=게시일로 서로 다를 수 있다.

### permalink — 반드시 명시

`_config.yml` 기본값(`/:categories/:title/`)은 한글 파일명 기반이라 URL이 길어진다.
**모든 포스트에 `permalink`를 명시**하고, 슬러그는 `ref`와 완전히 일치시킨다.

- 한국어: `/<카테고리 영문>/<ref>/` — `생각`→`thoughts`, `작업`→`work`, `개발`→`dev`
- 영어: `/en/<ref>/`
- 끝 슬래시 필수. 소문자 + 하이픈, ASCII만.

### categories / tags / ref

- 카테고리는 **항상 하나만**. 에세이·사유 = `생각`(영문판 `thoughts`),
  작업물 제작기·기술 튜토리얼 = `작업`(`work`), 코드 재현·툴 개발기 = `개발`(`dev`).
- 태그는 2~3개(최대 4). 최근 글은 `AI`가 기본값 격. 영문판 태그는 한국어 태그를
  같은 개수·같은 순서로 번역(라틴 문자 태그는 그대로 통과).
- `ref`는 제목의 **의미 번역** 슬러그(음차 아님), 관사·전치사 생략, 3~5단어.
  예: `현존의 아우라` → `aura-of-presence`.

## 2. 본문 문체 (한국어)

- **평서체 `~다`로 통일.** `~요`/`~습니다` 금지. 1인칭 "나는" 자연스럽게 사용.
  권유(`예를 들어 보자`), 반문(`~아닐까?`), 유보(`~일지도 모른다`, `~에 가깝다`)가
  저자의 어미 레퍼토리.
- **분량**: `생각` 에세이는 2,000~4,500자. 기술 글은 더 길어도 된다.
- **도입**: 인사·예고·요약 없이 프론트매터 뒤 빈 줄 하나, 곧바로 본론 첫 문장.
  첫 문단이 목록 페이지의 발췌문(excerpt)이 되므로 혼자 서는 문단으로 쓴다.
- **마무리**: "정리하면" 식 요약 금지. 격언형 한 문장, 열린 질문, 또는 농담으로 닫는다.
- **문단**: 2~5문장, 빈 줄 구분. **한 문장 문단을 전환점으로 꽂는 것**이 저자의 습관
  (예: "그런 의미에서 AI는 학습의 장벽을 허물지 않는다."). 질문 연쇄 리듬도 즐겨 쓴다
  ("픽셀 공간? RGB 공간? HSV나 HSL 공간?").
- **`생각` 에세이에는 헤더·리스트·볼드를 쓰지 않는다.** 강조는 서식이 아니라 구조
  (문단 분리, 번호 목록화)로 한다. 명시 요청 없이 볼드·이탤릭을 더하지 말 것.
- 헤더는 제작기·튜토리얼·열거형·연대기 글에만. **닫힌 ATX 스타일** `## 제목 ##`,
  H2/H3만 사용(H1 금지 — 제목은 레이아웃이 렌더).
- 열거가 필요하면 리스트 대신 `### 1. 소제목 ###` 헤더나 `**1.**` 볼드 번호를 쓴다.
  단, 글 도입에서 논지를 2개로 선언할 때는 번호 목록 허용.
- **부호 습관**: 인용·개념 지칭은 곧은 홑따옴표 `'…'`(컬리 인용부호 금지).
  원어 병기는 붙여 쓴 괄호 — `보간(interpolation)`, 인물은 `카파시(Andrej Karpathy)`,
  첫 등장 시 1회만. em-dash는 제목 부제·캡션에만 쓰고 본문 삽입절은 쉼표나 괄호로.
  작품명은 `〈머니볼〉`, 병렬은 중점 `·`, 범위는 `3~4배`.
- 비유·통계의 한계를 스스로 깎는 유보 문장을 말미에 넣는 습관이 있다
  ("물론 이 계산은 진지한 통계라기보다 수학적 농담에 가깝다.").
- 인스타/페이스북에서 옮겨온 글만 첫 줄에 `> 2026년 4월에 인스타 스토리에 쓴 글을 옮겨왔습니다.`
  인용문을 넣는다. 처음부터 블로그용으로 쓰는 글에는 넣지 않는다.

## 3. 이미지·미디어

### figure — 현행 표준

```html
<figure>
<img src="/assets/2026-06-19-love-and-slot-machine/slot-machine.jpg" alt="1899년 찰스 페이가 만든 최초의 슬롯머신 Liberty Bell">
<figcaption markdown="span">1899년 찰스 페이가 만든 최초의 슬롯머신 Liberty Bell. 한쪽에 달린 레버가 팔처럼 보여 one-armed bandit이라 불렸다. 출처: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Liberty_bell.jpg)</figcaption>
</figure>
```

- 들여쓰기 없음, 앞뒤 빈 줄. `figcaption`의 `markdown="span"`은 캡션 안 링크·수식
  렌더링에 필수.
- 캡션에 "그림 N"을 직접 쓰지 않는다 — CSS가 `Figure N.`을 자동으로 붙인다
  (`assets/css/style.css`).
- 캡션 형식: `설명 문장. 출처: [사이트명](URL)`. 논문 그림은 `출처: [저장소](…) · 논문: [저자, 연도](arxiv)`.
  자체 제작 그림은 출처 생략. `alt`는 항상 서술형으로 채운다.

### 에셋

- 폴더: `assets/YYYY-MM-DD-<짧은-영어-슬러그>/`, 파일명은 소문자 영어(`slot-machine.jpg`).
- 포맷: 다이어그램 = SVG, 사진·렌더 = JPG, 스크린샷·비교 이미지 = PNG, 애니메이션 = GIF.
- **한/영 포스트는 같은 에셋 파일을 공유**한다. 영어용 사본을 만들지 않는다.
- 경로는 루트 상대(`/assets/...`). GitHub raw URL은 구식이므로 쓰지 않는다.

### 임베드

- 유튜브: 공유 버튼이 뱉는 `<iframe width="560" height="315" …>`를 래퍼 없이 그대로.
  반응형은 CSS가 처리. 영상 작업 글은 iframe이 본문 첫 요소.
- 인스타그램: `<blockquote class="instagram-media" data-instgrm-permalink="…" data-instgrm-version="14"></blockquote>` + `<script async src="//www.instagram.com/embed.js"></script>` 쌍.
- 셰이더 데모는 iframe이 아니라 **정지 렌더 이미지 + Shadertoy 링크**로.
- 코드 블록은 언어 태그 필수(``` ```glsl ``` 등), 함수 단위로 쪼개고 사이에 산문 설명.

## 4. 각주와 출처

kramdown 각주. 기본은 각주 없음 — 실증 주장·논문·통계 인용 시에만 쓴다(1~3개가 보통).

```markdown
예술은 아름다움을 느끼게 한다[^zeki].

[^zeki]: Zeki et al., ["The experience of mathematical beauty and its neural correlates"](https://www.frontiersin.org/articles/10.3389/fnhum.2014.00068/full), *Frontiers in Human Neuroscience* (2014).
```

- 마커는 숫자보다 **의미형 키**(`[^zeki]`, `[^einstein]`)가 최신 관례. 정의는 글 맨 아래에 모은다.
- 형식: `저자, ["논문 제목"](링크), *학술지 이탤릭* 권(호), 쪽수 (연도). 한국어 부연.`
  부연은 근거의 한계를 스스로 밝히는 유보가 특징. 복수 링크는 `·`로 연결.
- 마커의 마침표 앞/뒤 위치는 글 안에서만 통일하면 된다.

## 5. 수식

MathJax v2(`_layouts/default.html`). **인라인 `$...$`, 블록 `$$...$$`만 쓴다.**
영문판에는 수식을 그대로 복사하고 주변 산문만 번역한다.

## 6. 포스트 간 링크

- 순수 마크다운 링크 + **대상 포스트의 `permalink` 절대 경로**. Liquid/`relative_url` 금지.
- 한국어 글 → `/thoughts/<ref>/`, 영어 글 → `/en/<ref>/`.
- **영문판에서는 내부 링크를 반드시 `/en/…`으로 재작성한다** (한국어 URL 복붙 금지).
- 주의: 링크는 파일명이 아니라 permalink 기준. 예외 사례 — `art-as-virus`(ref)의
  실제 URL은 `/thoughts/art-and-virus/`, `/en/art-and-virus/`.
- 선행 글 참조는 도입 문단에서 자연스럽게: "얼마 전 나는 ['공간에 대한 생각'](/thoughts/thoughts-on-space/)이라는 제목의 글을 썼다."

## 7. 영어 번역 규칙

- **문단 1:1 대응, 문장 단위로 충실하게** — 내용을 더하거나 빼지 않는다. 헤딩·리스트·
  인용·코드·figure 위치도 그대로.
- 문체는 자연스러운 영어 에세이체. 축약형(don't, it's) 자유롭게, 한국어의 반문은
  영어에서도 질문으로 유지.
- 제목은 직역이 아니라 **의역 + Title Case** (`똑똑함의 정의` → `What It Means to Be Smart`).
- 한국 특유의 인물·밈·기관은 각주 없이 **인라인으로 짧게 gloss**한다
  (`이동진 평론가` → `film critic Lee Dong-jin`). 한국어 문맥이 전제한 "국내"는
  "here in Korea"처럼 명시. 인명은 개정 로마자 + 하이픈(`Lee Jae-yong`),
  공식 표기가 있으면 그것(`IRENE`, `Jang Wonyoung`).
- 캡션·alt 모두 번역. `출처:` → `Source:`, 링크 URL은 그대로. 코드는 그대로 두고
  **주석만 번역**. 이미지·iframe URL은 바이트 단위로 동일하게.
- 각주 키는 한/영 동일, 서지 인용은 그대로, 해설 문장만 번역.
- 인용부호는 곧은 큰따옴표 `"…"`.

## 8. 커밋

요청받았을 때만 커밋/푸시한다. 기본 브랜치는 `master`.

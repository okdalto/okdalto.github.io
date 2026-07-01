# okdalto.github.io — 작업 규칙

Jekyll 기반 한/영 이중 언어 블로그.

## 새 포스트 작성 규칙

포스트는 **한국어(`_posts/`) + 영어(`_en/`)** 쌍으로 만든다.

1. **한국어 원문**: `_posts/YYYY-MM-DD-<한글제목>.md`
2. **영어 번역**: `_en/<english-slug>.md` (파일명 = 영어 슬러그 = `ref` 값)
3. 두 파일의 프론트매터에 **같은 `ref:`** 값을 넣는다. 언어 전환 버튼은 `ref`로 짝을 찾는다(`_layouts/default.html`). URL과 무관하게 동작하므로 permalink를 바꿔도 안전하다.

### 프론트매터 예시

한국어 (`_posts/`):

```yaml
---
title: "예술과 바이러스"
permalink: /thoughts/art-and-virus/
date: 2026-07-01T00:00:00+09:00
categories:
  - 생각
tags:
  - 예술
ref: art-as-virus
---
```

영어 (`_en/`):

```yaml
---
title: "Art and Virus"
permalink: /en/art-and-virus/
date: 2026-07-01T00:00:00+09:00
categories:
  - thoughts
tags:
  - art
ref: art-as-virus
---
```

## Permalink(URL) 규칙 — 중요

`_config.yml`의 기본 permalink는 `/:categories/:title/`인데, 이는 파일명(긴 한글 제목)과
**모든 카테고리**를 이어붙여 URL이 매우 길어진다. 그래서 **포스트마다 `permalink:`를 명시**해
짧은 ASCII URL로 고정한다.

- 한국어 포스트: `permalink: /<카테고리(영문)>/<ref>/`
- 영어 포스트: `permalink: /en/<ref>/`
- 카테고리 → 영문 매핑 (첫 번째 카테고리 기준):
  - `생각` → `thoughts`
  - `작업` → `work`
  - `개발` → `dev`
- 슬러그는 `ref` 값을 그대로 사용한다(모두 소문자, 하이픈).

> 참고: `_config.yml`의 전역 규칙만으로는 한글 제목을 자동으로 짧은 영어 URL로 바꿀 수
> 없다(Jekyll의 `:title`/`:slug`은 파일명 기반). 그래서 포스트별 `permalink` 명시가 관례다.

## 수식

MathJax v2 사용(`_layouts/default.html`). 인라인 `$...$` 또는 `\(...\)`, 블록 `$$...$$`.

## 날짜

미래 날짜 포스트는 GitHub Pages가 게시하지 않으므로, 게시 시점보다 앞서지 않게 한다.

## 커밋

요청받았을 때만 커밋/푸시한다. 기본 브랜치는 `master`.

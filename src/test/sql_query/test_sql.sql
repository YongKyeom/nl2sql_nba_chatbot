
-- 테이블 리스트
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name;

-- 테이블 상세 스키마
PRAGMA table_info(game);
SELECT sql FROM sqlite_master WHERE type='table' AND name='game';

SELECT * FROM game LIMIT 1;
SELECT * FROM play_by_play LIMIT 1;
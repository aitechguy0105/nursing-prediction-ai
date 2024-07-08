SELECT *,
CASE WHEN path_info like '/login' THEN 'login'
WHEN path_info like '/admin%' THEN 'admin'
WHEN path_info like '%release-hijack%' THEN 'did-hijack'
WHEN path_info like '%hijack/%' THEN 'was-hijacked'
ELSE 'unknown'
END AS new_pathinfo, 1 as counter
from axes_accesslog aa where attempt_time >  CURRENT_DATE - INTERVAL '3 months'
order by attempt_time desc
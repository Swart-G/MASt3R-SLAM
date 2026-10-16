# MASt3R-SLAM wire protocol

`mast3r_slam.proto` is the single source of truth for protocol version 2. Both
the Python server and Android client generate bindings from this file.

The WebSocket transports one binary, length-independent Protobuf message per
WebSocket frame. The first client message must be `Hello`. Large point-cloud
chunks are deliberately excluded from the socket and downloaded from the HTTPS
URLs advertised by `CloudManifest`.

Version 2 adds latest-only `ArAlignmentFrame` messages and
`ArAlignmentResult` (`SEARCHING`/`ALIGNED`/`ALIGNMENT_LOST`) so the ARCore world
is registered to the immutable SLAM map before any mesh is drawn.

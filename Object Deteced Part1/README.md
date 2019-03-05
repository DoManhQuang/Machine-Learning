## Vấn đề xử lý của bài toán.

Trong xử lý ảnh sử dụng tính đặc trưng (histogram) của màu sắc 
để xác định đối tượng cũng sẽ gây ra sai lệch rất nhiều. Đặc 
biệt đối với các đối tượng cùng màu sắc nhưng khác loài gần 
 như là tương đồng nhau, sẽ gây ra sự nhầm lẫn .
 
## Một số trường hợp hay gặp.
 
 Mèo có màu lông đen sẽ giống Chó có màu lông đen nhiều hơn là 
 giống Mèo màu lông vàng và ngược lại.
 
## Tại sao lại vậy ?
 
 Do chúng ta chỉ trích chọn đặc trưng của bức ảnh nên dẫn đến sai 
 lệch.
 
## Giải pháp.
Một Giải pháp khá ổn đó là chúng ta sẽ tăng dữ liệu kiểm tra ở 
dataset , chúng ta lập ra được biểu đồ về sự giống nhau giữa các 
bức ảnh đầu vào với bức ảnh có sẵn trong dataset.

# Chart Cat 
 [0.946, 0.829, 0.143, 0.947, 0.052, 0.252]

# Chart Dog 
 [0.977, 0.772, 0.371, 0.966, 0.331, 0.509]

## Tính kỳ vọng.
Biểu đồ cho ta thấy rõ sự khác nhau và giống nhau giữa các màu sắc 
của ảnh.Cuối cùng thì ta sẽ tính kỳ vọng của ảnh để đưa ra kết quả.
      
## Help
Nếu bạn gặp vấn đề gì đó hãy liên hệ lại với [Tôi](https://github.com/DoManhQuang) or reach out on facebook [@Đỗ Mạnh Quang](https://www.facebook.com/ManhQuangITBlue) .

 
